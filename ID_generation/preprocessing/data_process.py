"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import ast
import gzip
import html
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import requests
import tqdm
import wget


def parse(path):  # for Amazon
    g = gzip.open(path, "r")
    for l in g:
        l = l.replace(b"true", b"True").replace(b"false", b"False")
        yield eval(l)


def download_file(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {os.path.basename(path)}")
    else:
        print(f"Failed to download {os.path.basename(path)}")


class Yelp:
    """
    Process LETTER-format Yelp data into liger pipeline format.
    Yelp data comes pre-processed from LETTER repo with:
    - Yelp.inter.json: User-item interaction sequences
    - Yelp.item.json: Item metadata (title, description/categories)
    """
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.data_path = os.path.join(root, "Yelp")
        
    def process(self):
        """
        Load Yelp.inter.json and convert to (user, item, timestamp) format.
        
        LETTER format: {"user_id": [item_id1, item_id2, ...]}
        Output: List of (user_id, item_id, timestamp) tuples
        
        Note: Since LETTER data doesn't include timestamps, we use sequential
        ordering (1, 2, 3, ...) to maintain interaction order.
        """
        inter_file = os.path.join(self.data_path, "Yelp.inter.json")
        
        if not os.path.exists(inter_file):
            raise FileNotFoundError(
                f"Yelp interaction file not found: {inter_file}\n"
                f"Please copy Yelp.inter.json from LETTER repo to {self.data_path}/"
            )
        
        print(f"Loading Yelp interactions from {inter_file}")
        with open(inter_file, 'r') as f:
            interactions = json.load(f)
        
        datas = []
        # LETTER format: {"user_id": [item1, item2, ...]}
        # Convert to (user, item, timestamp) format with sequential timestamps
        for user_id, items in interactions.items():
            # Use sequential timestamps to maintain order
            for idx, item_id in enumerate(items):
                timestamp = idx + 1  # Sequential ordering starting from 1
                # Keep user_id and item_id as strings (will be mapped by id_map later)
                datas.append((str(user_id), str(item_id), timestamp))
        
        print(f"Loaded {len(datas)} interactions from {len(interactions)} users")
        return datas
    
    def process_meta(self, data_maps):
        """
        Load Yelp.item.json and create metadata dict.
        
        LETTER format: {"item_id": {"title": "...", "description": "..."}}
        Output: Dict of {item_id: metadata_dict}
        """
        item_file = os.path.join(self.data_path, "Yelp.item.json")
        
        if not os.path.exists(item_file):
            raise FileNotFoundError(
                f"Yelp item file not found: {item_file}\n"
                f"Please copy Yelp.item.json from LETTER repo to {self.data_path}/"
            )
        
        print(f"Loading Yelp item metadata from {item_file}")
        with open(item_file, 'r') as f:
            items_raw = json.load(f)
        
        datas = {}
        item_ids = set(data_maps["item2id"].keys())
        
        for item_id_str, meta in items_raw.items():
            if item_id_str not in item_ids:
                continue
            
            # LETTER format: {"title": "Business Name", "description": "Cat1, Cat2, ..."}
            # Convert to format compatible with liger's meta_map
            datas[item_id_str] = {
                "title": meta.get("title", "Unknown Business"),
                "description": meta.get("description", ""),
                # Parse categories from comma-separated description
                "categories": [cat.strip() for cat in meta.get("description", "").split(", ") if cat.strip()]
            }
        
        print(f"Loaded metadata for {len(datas)} items")
        return datas


class Yelp_oracle(Yelp):
    """
    Yelp dataset with oracle date-category context.
    Extends Yelp to:
    1. Load real unix timestamps (from Yelp.inter_timestamps.json)
    2. Auto-generate oracle date-category context JSONL from item metadata
    """

    def __init__(self, root):
        super().__init__(root)

    def process(self):
        """
        Load Yelp.inter.json with REAL timestamps from Yelp.inter_timestamps.json.
        Falls back to sequential if timestamps file not found.
        """
        inter_file = os.path.join(self.data_path, "Yelp.inter.json")
        timestamps_file = os.path.join(self.data_path, "Yelp.inter_timestamps.json")

        if not os.path.exists(inter_file):
            raise FileNotFoundError(
                f"Yelp interaction file not found: {inter_file}\n"
                f"Please run preprocess_yelp.py to generate Yelp data files."
            )

        print(f"Loading Yelp_oracle interactions from {inter_file}")
        with open(inter_file, 'r') as f:
            interactions = json.load(f)

        all_timestamps = None
        if os.path.exists(timestamps_file):
            print(f"Loading real timestamps from {timestamps_file}")
            with open(timestamps_file, 'r') as f:
                all_timestamps = json.load(f)
        else:
            print(f"WARNING: {timestamps_file} not found, using sequential timestamps.")

        datas = []
        for user_id, items in interactions.items():
            for idx, item_id in enumerate(items):
                if all_timestamps is not None and user_id in all_timestamps:
                    timestamp = int(all_timestamps[user_id][idx])
                else:
                    timestamp = idx + 1
                datas.append((str(user_id), str(item_id), timestamp))

        ts_type = 'real timestamps' if all_timestamps is not None else 'sequential timestamps'
        print(f"Loaded {len(datas)} interactions from {len(interactions)} users ({ts_type})")
        return datas

    def generate_oracle_context(self, data_maps, output_path):
        """
        Generate oracle date-category context JSONL from item metadata.
        For each unique date in the dataset, compute top-5 categories by lift
        (observed share / baseline share) and format as context text.

        Output format (compatible with load_date_context()):
        {"recordId": "2019-01-01", "modelOutput": {"content": [{"text": "today is 2019-01-01, likely categories: Cat1, Cat2, ..."}]}}
        """
        # Load item metadata for categories
        item_file = os.path.join(self.data_path, "Yelp.item.json")
        if not os.path.exists(item_file):
            raise FileNotFoundError(f"Yelp item file not found: {item_file}")

        with open(item_file, 'r') as f:
            items_raw = json.load(f)

        # Build item_id -> categories mapping (using raw item IDs before id_map)
        item_categories = {}
        for item_id_str, meta in items_raw.items():
            cats = [c.strip() for c in meta.get("description", "").split(",") if c.strip()]
            # Filter out yelp/event categories
            cats = [c for c in cats if not any(kw in c.lower() for kw in ["yelp", "event", "elite"])]
            item_categories[item_id_str] = cats

        # Load interactions + timestamps to compute per-date category counts
        inter_file = os.path.join(self.data_path, "Yelp.inter.json")
        timestamps_file = os.path.join(self.data_path, "Yelp.inter_timestamps.json")

        with open(inter_file, 'r') as f:
            interactions = json.load(f)

        all_timestamps = None
        if os.path.exists(timestamps_file):
            with open(timestamps_file, 'r') as f:
                all_timestamps = json.load(f)

        # Count categories per date and overall baseline
        date_cat_counts = defaultdict(lambda: defaultdict(int))
        date_totals = defaultdict(int)
        cat_overall = defaultdict(int)
        overall_total = 0

        for user_id, items in interactions.items():
            for idx, item_id in enumerate(items):
                item_id_str = str(item_id)
                cats = item_categories.get(item_id_str, [])
                if not cats:
                    continue

                if all_timestamps is not None and user_id in all_timestamps:
                    ts = int(all_timestamps[user_id][idx])
                    date_key = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                else:
                    date_key = f"seq-{idx}"  # fallback

                for cat in cats:
                    date_cat_counts[date_key][cat] += 1
                    cat_overall[cat] += 1
                date_totals[date_key] += 1
                overall_total += 1

        # Compute baseline share
        cat_baseline = {cat: cnt / overall_total for cat, cnt in cat_overall.items()}

        # For each date, rank categories by lift, take top 5
        MIN_COUNT = 3
        TOP_K = 5

        lines = []
        for date_key in sorted(date_cat_counts.keys()):
            dtotal = date_totals[date_key]
            if dtotal == 0:
                continue
            cat_lifts = []
            for cat, count in date_cat_counts[date_key].items():
                if count < MIN_COUNT:
                    continue
                share = count / dtotal
                base = cat_baseline.get(cat, 0)
                if base > 0:
                    lift = share / base
                    cat_lifts.append((cat, lift))
            cat_lifts.sort(key=lambda x: x[1], reverse=True)
            top_cats = [cat for cat, _ in cat_lifts[:TOP_K]]

            if top_cats:
                text = f"today is {date_key}, likely categories: {', '.join(top_cats)}"
                record = {
                    "recordId": date_key,
                    "modelOutput": {"content": [{"text": text}]}
                }
                lines.append(json.dumps(record))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        print(f"Generated oracle context for {len(lines)} dates → {output_path}")
        return output_path


class Yelp_daytime(Yelp_oracle):
    """
    Yelp dataset with fine-grained daytime context (morning/afternoon/evening/night).
    Extends Yelp_oracle to convert unix timestamps to local time using business
    state → UTC offset, then bin into time-of-day periods.
    """

    STATE_TO_UTC_OFFSET = {
        "AL": -6, "CT": -5, "DE": -5, "FL": -5, "GA": -5, "IN": -5,
        "KY": -5, "MA": -5, "MD": -5, "ME": -5, "MI": -5, "NC": -5,
        "NH": -5, "NJ": -5, "NY": -5, "OH": -5, "PA": -5, "RI": -5,
        "SC": -5, "TN": -6, "VA": -5, "VT": -5, "WV": -5,
        "AR": -6, "IA": -6, "IL": -6, "KS": -6, "LA": -6, "MN": -6,
        "MO": -6, "MS": -6, "ND": -6, "NE": -6, "OK": -6, "SD": -6,
        "TX": -6, "WI": -6,
        "AZ": -7, "CO": -7, "ID": -7, "MT": -7, "NM": -7, "UT": -7, "WY": -7,
        "CA": -8, "NV": -8, "OR": -8, "WA": -8,
        "AK": -9, "HI": -10,
        "AB": -7, "BC": -8, "MB": -6, "NB": -4, "NL": -4, "NS": -4,
        "NT": -7, "ON": -5, "PE": -4, "QC": -5, "SK": -6, "YT": -8,
    }
    DEFAULT_UTC_OFFSET = -6

    TIME_BINS = {
        "morning":   (6, 11),
        "afternoon": (11, 17),
        "evening":   (17, 22),
        "night":     (22, 6),
    }

    def __init__(self, root, business_file=None):
        super().__init__(root)
        self.business_file = business_file
        self.biz_offset = {}  # business_id → UTC offset hours

    def load_business_offsets(self, item_ids=None):
        """Load business_id → UTC offset from Yelp business dataset."""
        if not self.business_file or not os.path.exists(self.business_file):
            print(f"WARNING: business_file not found: {self.business_file}")
            return
        print(f"Loading business timezone offsets from {self.business_file}")
        with open(self.business_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                bid = r.get("business_id", "")
                if item_ids is not None and bid not in item_ids:
                    continue
                state = (r.get("state", "") or "").strip().upper()
                self.biz_offset[bid] = self.STATE_TO_UTC_OFFSET.get(
                    state, self.DEFAULT_UTC_OFFSET
                )
        print(f"  Loaded offsets for {len(self.biz_offset)} businesses")

    @staticmethod
    def _get_time_bin(hour):
        for name, (start, end) in Yelp_daytime.TIME_BINS.items():
            if start < end:
                if start <= hour < end:
                    return name
            else:  # wraps midnight
                if hour >= start or hour < end:
                    return name
        return "night"

    def compute_daytime_keys(self, user_items_id, user_timestamps_id, data_maps):
        """
        Compute per-interaction daytime context keys: "YYYY-MM-DD_timebin".
        
        Args:
            user_items_id: {mapped_user_id: [mapped_item_id, ...]}
            user_timestamps_id: {mapped_user_id: [unix_ts, ...]}
            data_maps: contains id2item for reverse mapping
            
        Returns:
            {mapped_user_id: ["2019-01-01_night", "2019-01-02_morning", ...]}
        """
        id2item = data_maps["id2item"]
        user_daytime_keys = {}

        for uid, items in user_items_id.items():
            ts_list = user_timestamps_id.get(uid, [])
            keys = []
            for item_id_str, ts in zip(items, ts_list):
                raw_item = id2item.get(item_id_str, item_id_str)
                offset = self.biz_offset.get(raw_item, self.DEFAULT_UTC_OFFSET)
                local_dt = datetime.utcfromtimestamp(int(ts)) + timedelta(hours=offset)
                date_key = local_dt.strftime("%Y-%m-%d")
                tbin = self._get_time_bin(local_dt.hour)
                keys.append(f"{date_key}_{tbin}")
            user_daytime_keys[uid] = keys

        return user_daytime_keys


class Amazon:
    def __init__(self, root, dataset_name, rating_score):
        self.root = os.path.abspath(root)
        self.dataset_name = dataset_name
        self.rating_score = rating_score
        self.download()
        self.process()

    def download(self):
        path = os.path.join(self.root, "amazon")
        os.makedirs(path, exist_ok=True)

        url_dict = {
            "Beauty": [
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
            ],
            "Toys_and_Games": [
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
            ],
            "Sports_and_Outdoors": [
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz",
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz",
            ],
        }
        # we save the raw data into `directory`
        for url in url_dict[self.dataset_name]:
            # Extract the filename from the URL
            filename = url.split("/")[-1]
            filepath = os.path.join(path, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found, downloading...")
                download_file(url, filepath)

    def process(self):
        """
        reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
        asin - ID of the product, e.g. 0000013714
        reviewerName - name of the reviewer
        helpful - helpfulness rating of the review, e.g. 2/3
        --"helpful": [2, 3],
        reviewText - text of the review
        --"reviewText": "I bought this for my husband who plays the piano. ..."
        overall - rating of the product
        --"overall": 5.0,
        summary - summary of the review
        --"summary": "Heavenly Highway Hymns",
        unixReviewTime - time of the review (unix time)
        --"unixReviewTime": 1252800000,
        reviewTime - time of the review (raw)
        --"reviewTime": "09 13, 2009"
        """
        datas = []
        data_file = os.path.join(
            self.root, "amazon", f"reviews_{self.dataset_name}_5.json.gz"
        )

        for inter in parse(data_file):
            if float(inter["overall"]) <= self.rating_score:
                continue
            user = inter["reviewerID"]
            item = inter["asin"]
            time = inter["unixReviewTime"]
            datas.append((user, item, int(time)))
        return datas

    def process_meta(self, data_maps):
        """
        asin - ID of the product, e.g. 0000031852
        --"asin": "0000031852",
        title - name of the product
        --"title": "Girls Ballet Tutu Zebra Hot Pink",
        description
        price - price in US dollars (at time of crawl)
        --"price": 3.17,
        imUrl - url of the product image (str)
        --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
        related - related products (also bought, also viewed, bought together, buy after viewing)
        --"related":{
            "also_bought": ["B00JHONN1S"],
            "also_viewed": ["B002BZX8Z6"],
            "bought_together": ["B002BZX8Z6"]
        },
        salesRank - sales rank information
        --"salesRank": {"Toys & Games": 211836}
        brand - brand name
        --"brand": "Coxlures",
        categories - list of categories the product belongs to
        --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
        """
        datas = {}
        meta_file = os.path.join(
            self.root, "amazon", f"meta_{self.dataset_name}.json.gz"
        )

        item_asins = set(data_maps["item2id"].keys())
        for info in parse(meta_file):
            if info["asin"] not in item_asins:
                continue
            datas[info["asin"]] = info
        return datas


class Steam:
    def __init__(self, root, user_core) -> None:
        self.root = os.path.abspath(root)
        self.urls = {
            "reviews": "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
            "games": "http://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
        }
        self.download()
        self.process(user_core)

    def download(self):
        path = os.path.join(self.root, "steam")

        if os.path.exists(os.path.join(path, "steam_reviews.json")) and os.path.exists(
            os.path.join(path, "steam_games.json")
        ):
            print(f"{path} exists, download is not needed.")
            return

        os.makedirs(path, exist_ok=True)
        for d in ["games", "reviews"]:
            print(f"downloading steam from {self.urls[d]}")
            file_name = wget.download(self.urls[d], out=path)
            content = gzip.open(file_name, "rb")
            content = content.read().decode("utf-8").split("\n")
            content = [
                json.loads(json.dumps(ast.literal_eval(line)))
                for idx, line in enumerate(content)
                if line
            ]
            with open(file_name[:-3], "w") as f:
                json.dump(content, f)

    def process(self, user_core):
        path = os.path.join(self.root.replace("raw_data", "processed"))
        os.makedirs(path, exist_ok=True)

        print(f"preprocessing steam ...")
        review_file = glob(f"{os.path.join(self.root, 'steam')}/steam_reviews.json")[0]

        with open(review_file, "r") as f:
            raw_data = json.load(f)

        user_counts = Counter([entry["username"] for entry in raw_data])
        raw_data = [
            entry for entry in raw_data if user_counts[entry["username"]] >= user_core
        ]

        user_id, item_id = 1, 1
        self.user2id, self.item2id, self.id2user, self.id2item = {}, {}, {}, {}
        self.item2id["<pad>"], self.id2item[0] = 0, "<pad>"
        self.item2review = {}
        for entry in tqdm.tqdm(raw_data, desc="Mapping unique users and items ..."):
            if entry["username"] not in self.user2id:
                self.user2id[entry["username"]] = user_id
                self.id2user[user_id] = entry["username"]
                user_id += 1
            if entry["product_id"] not in self.item2id:
                self.item2id[entry["product_id"]] = item_id
                self.id2item[item_id] = entry["product_id"]
                self.item2review[item_id] = entry["text"]
                item_id += 1

        self.sequence_raw = []
        for entry in tqdm.tqdm(raw_data, desc="Constructing sequence and graph ..."):
            self.sequence_raw.append(
                (
                    self.user2id[entry["username"]],
                    self.item2id[entry["product_id"]],
                    int(datetime.fromisoformat(entry["date"]).timestamp()),
                )
            )
        # the item in sequence_raw is in item_id domain.

    def process_meta_infos(self):
        meta_file = glob(f"{os.path.join(self.root, 'steam')}/steam_games.json")[0]
        with open(meta_file, "r") as f:
            raw_data = json.load(f)

        items = {}
        for entry in tqdm.tqdm(raw_data, desc="Creating item content features...."):
            if "id" in entry and entry["id"] in self.item2id:
                meta_dict = {}
                meta_dict["title"] = f"{entry['title']}"
                meta_dict["genre"] = (
                    f"{' '.join(entry['genres']) if 'genres' in entry else 'Unknown'}"
                )
                meta_dict["tags"] = (
                    f"{' '.join(entry['tags']) if 'tags' in entry else 'Unknown'}"
                )
                meta_dict["specs"] = (
                    f"{' '.join(entry['specs']) if 'specs' in entry else 'Unknown'}"
                )
                meta_dict["price"] = f"{entry.get('price', 0)}"
                meta_dict["publisher"] = f"{entry.get('publisher', 'Unknown')}"
                meta_dict["sentiment"] = f"{entry.get('sentiment', 'Unknown')}"
                items[self.item2id[entry["id"]]] = (
                    meta_dict  # id is product_id, item2id maps product_id to item_id that start from 1
                )

        return items


def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True


def filter_Kcore(user_items, user_core, item_core):
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    # user_count: number of item this user has interacted with
    # item_count: number of user this item has appeared
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:  # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def get_attribute_Amazon(meta_infos, datamaps, attribute_core):
    category_key = "categories"

    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        for cates in info[category_key]:
            for cate in cates[1:]:
                attributes[cate] += 1
        try:
            attributes[info["brand"]] += 1
        except:
            pass

    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []

        try:
            if attributes[info["brand"]] >= attribute_core:
                new_meta[iid].append(info["brand"])
        except:
            pass
        for cates in info[category_key]:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)

    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, item_attributes in new_meta.items():
        item_id = datamaps["item2id"][iid]
        items2attributes[item_id] = []
        for attribute in item_attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(
        f"attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}"
    )
    datamaps["attribute2id"] = attribute2id
    datamaps["id2attribute"] = id2attribute
    datamaps["attributeid2num"] = attributeid2num
    return (
        len(attribute2id),
        np.mean(attribute_lens),
        datamaps,
        items2attributes,
        attributes,
    )


def get_attribute_steam(meta_infos, datamaps, attribute_core):
    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        try:
            attributes[info["genre"]] += 1
        except:
            pass
    print(f"before delete, attribute num:{len(attributes)}")
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []

        try:
            if attributes[info["genre"]] >= attribute_core:
                new_meta[iid].append(info["genre"])
        except:
            pass

    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in new_meta.items():
        try:
            item_id = datamaps["item2id"][iid]
        except:
            continue

        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))

    print(f"after delete, attribute num:{len(attribute2id)}")
    print(
        f"attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}"
    )
    datamaps["attribute2id"] = attribute2id
    datamaps["id2attribute"] = id2attribute
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def clean_text2(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = " ".join(raw_text[0])
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."
    if len(cleaned_text) >= 2000:
        cleaned_text = ""
    return cleaned_text


def meta_map(
    meta_infos,
    item2id,
    attributes,
    features_needed=["title", "price", "brand", "categories", "description"],
    attribute_core=0,
    prompt_format="v1",
):
    id2meta = {}
    item2meta = {}

    for item, meta in tqdm.tqdm(meta_infos.items()):
        meta_text = ""
        keys = set(meta.keys()).intersection(features_needed)

        if prompt_format == "amazon":
            if "title" in keys and meta["title"] != "":
                meta_text += f"This item is called '{feature_process(meta['title'])}'. "
            if "price" in keys:
                meta_text += f"It is priced at {feature_process(meta['price'])} "
            if (
                "brand" in keys
                and meta["brand"] != ""
                and attributes[meta["brand"]] >= attribute_core
            ):
                meta_text += (
                    f"It is manufactured by '{feature_process(meta['brand'])}'. "
                )
            if "categories" in keys:
                meta_text += f"It belongs to the categories of {feature_process(meta['categories'])}. "
            if "description" in keys:
                meta_text += f"The description of this item is: {feature_process(meta['description'])}. "
        elif prompt_format == "unisrec":
            for meta_key in features_needed:
                if meta_key in meta:
                    meta_value = clean_text2(meta[meta_key])
                    meta_text += meta_value + " "

        elif prompt_format == "steam":
            # Title
            if "title" in keys and meta["title"] != "":
                meta_text += f"This game is called '{meta['title']}'. "
            if "publisher" in keys and meta["publisher"] != "":
                meta_text += f"Developed by {meta['publisher']}, "
            if "price" in keys and meta["price"] is not None:
                meta_text += f"this game is priced at {meta['price']}. "
            if "specs" in keys:
                specs_list = meta["specs"].split(" ")
                formatted_specs = ", ".join(specs_list)
                meta_text += f"This game features a variety of gameplay modes including {formatted_specs}. "
            if "sentiment" in keys:
                meta_text += (
                    f"The game holds a {meta['sentiment']} sentiment among players. "
                )
            if "tags" in keys:
                meta_text += f"Tags of the game include: {meta['tags']}. "
        
        elif prompt_format == "yelp":
            # Yelp business format
            if "title" in keys and meta["title"] != "":
                meta_text += f"This business is '{meta['title']}'. "
            if "description" in keys and meta["description"] != "":
                meta_text += f"Categories: {meta['description']}. "

        if (
            prompt_format == "steam"
        ):  # for steam, the key of the meta_infos are already passed in to item2id, so it is essentially id
            try:
                id = item2id[item]
            except:
                continue
            id2meta[id] = meta_text
        else:
            item2meta[item] = meta_text
            id = item2id[item]
            id2meta[id] = meta_text
    return id2meta


def get_item_review_map(review_mapping, data_maps, meta_infos):
    id2review = defaultdict(dict)
    for reviewer_id, items in review_mapping.items():
        try:
            user_id = data_maps["user2id"][reviewer_id]
            for item, review in items.items():
                id = data_maps["item2id"][item]
                title = (
                    "" if "title" not in meta_infos[item] else meta_infos[item]["title"]
                )
                categories = (
                    meta_infos[item]["categories"]
                    if "categories" in meta_infos[item]
                    else ""
                )
                id2review[user_id][id] = (
                    title,
                    categories,
                ) + review
        except:
            pass

    return id2review


def get_steam_reviews(user_seq, review_mapping, datamaps, meta_infos):
    id2review = defaultdict(list)
    for reviewer_id in user_seq.keys():
        user_id = datamaps["user2id"][reviewer_id]
        for item in user_seq[reviewer_id]:
            id = datamaps["item2id"][item]
            title = "" if "title" not in meta_infos[item] else meta_infos[item]["title"]
            genre = meta_infos[item]["genre"] if "genre" in meta_infos[item] else ""
            tags = meta_infos[item]["tags"] if "tags" in meta_infos[item] else ""
            genre = meta_infos[item]["specs"] if "specs" in meta_infos[item] else ""
            price = meta_infos[item]["price"] if "price" in meta_infos[item] else ""
            publisher = (
                meta_infos[item]["publisher"] if "publisher" in meta_infos[item] else ""
            )
            sentiment = (
                meta_infos[item]["sentiment"] if "sentiment" in meta_infos[item] else ""
            )
            id2review[user_id].append(
                {
                    "itemid": id,
                    "userid": user_id,
                    "title": title,
                    "tags": tags,
                    "genre": genre,
                    "price": price,
                    "publisher": publisher,
                    "sentiment": sentiment,
                    "review": review_mapping[item],
                }
            )
    return id2review


def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ""
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num) - i - 1) % 3 == 0:
            res_num += ","
    return res_num[:-1]


def id_map(user_items, user_timestamps=None):  # user_items dict

    user2id = {}  # raw 2 uid
    item2id = {}  # raw 2 iid
    id2user = {}  # uid 2 raw
    id2item = {}  # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    user_timestamps_id = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = []  # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
        if user_timestamps:
            user_timestamps_id[uid] = user_timestamps[user]
    data_maps = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
    }
    return final_data, user_id - 1, item_id - 1, data_maps


def get_interaction(datas, meta_data_set=None):
    user_seq = {}
    user_timestamps = {}  # NEW: store timestamps
    for data in datas:
        user, item, time = data
        if meta_data_set is not None and item not in meta_data_set:
            continue
        if user not in user_seq:
            user_seq[user] = []
            user_timestamps[user] = []  # NEW
        user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        items = []
        timestamps = []  # NEW
        for t in item_time:
            items.append(t[0])
            timestamps.append(t[1])  # NEW: keep timestamp
        user_seq[user] = items
        user_timestamps[user] = timestamps  # NEW
    return user_seq, user_timestamps  # MODIFIED return


def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(", ".join(l))
    else:
        return l


def clean_text(raw_text):
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    return text


def feature_process(feature):
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += "."
    elif len(feature) > 0 and isinstance(feature[0], list):
        # this should be the categories
        for v1 in feature:
            for v in v1[1:]:
                sentence += clean_text(v)
                sentence += ", "
        sentence = sentence[:-2]
        # sentence += '.'
    elif isinstance(feature, list):
        for v1 in feature:
            sentence += clean_text(v1)
    else:
        sentence = clean_text(feature)
    return sentence


def preprocessing(config):
    dataset_name = config["name"]
    data_file, id2meta_file, item2attribute_file, user_timestamps_map = preprocessing_each_dataset(
        config, dataset_name
    )
    return data_file, id2meta_file, item2attribute_file, user_timestamps_map


def preprocessing_each_dataset(config, dataset_name):
    data_type, features_needed = config["type"], config["features_needed"]
    prompt_format = config["prompt_format"]
    features_used = "_".join(features_needed)
    raw_data_path = config["raw_data_path"]
    # by default this is "./ID_generation/preprocessing/raw_data/"
    processed_data_path = config["processed_data_path"]
    # by default this is "./ID_generation/preprocessing/processed/"

    # a list of file path we will use
    data_file = os.path.join(processed_data_path, f"{dataset_name}.txt")
    id2meta_file = os.path.join(
        processed_data_path,
        f"{dataset_name}_{features_used}_{prompt_format}_id2meta.json",
    )
    item2attributes_file = os.path.join(
        processed_data_path, f"{dataset_name}_item2attributes.json"
    )

    print(f"data_name: {dataset_name}, data_type: {data_type}")

    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0
    print(f"Use user core: {user_core}, item core: {item_core}.")

    if data_type == "Amazon":
        dataset = Amazon(raw_data_path, dataset_name, rating_score)
        datas = dataset.process()
    elif data_type == "steam":
        dataset = Steam(raw_data_path, user_core)
        datas = dataset.sequence_raw
    elif data_type == "Yelp":
        dataset = Yelp(raw_data_path)
        datas = dataset.process()
    elif data_type == "Yelp_oracle":
        dataset = Yelp_oracle(raw_data_path)
        datas = dataset.process()
        # Auto-generate oracle context JSONL if not present
        oracle_context_path = os.path.join(raw_data_path, "Yelp", "Yelp_oracle_date_context.jsonl")
        if not os.path.exists(oracle_context_path):
            print("Oracle context not found, generating...")
            dataset.generate_oracle_context(None, oracle_context_path)
    elif data_type == "Yelp_daytime":
        business_file = config.get("business_file", None)
        dataset = Yelp_daytime(raw_data_path, business_file=business_file)
        datas = dataset.process()
    else:
        raise NotImplementedError

    meta_data_set = None
    user_items, user_timestamps = get_interaction(datas, meta_data_set)
    # raw_id user: [item1, item2, item3...]

    # filter K-core
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_timestamps = {u: user_timestamps[u] for u in user_items.keys()}
    print(f"User {user_core}-core complete! Item {item_core}-core complete!")
    user_items_id, user_num, item_num, data_maps = id_map(user_items, user_timestamps)
    user_timestamps_id = {
        data_maps["user2id"][user]: user_timestamps[user] 
        for user in user_items.keys()
    }
    user_count, item_count, _ = check_Kcore(
        user_items_id, user_core=user_core, item_core=item_core
    )

    # calculate sparsity
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = (
        np.mean(user_count_list),
        np.min(user_count_list),
        np.max(user_count_list),
    )
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = (
        np.mean(item_count_list),
        np.min(item_count_list),
        np.max(item_count_list),
    )
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    seqs_length = [len(user_items_id[i]) for i in user_items_id.keys()]
    show_info = (
        f"Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n"
        + f"Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n"
        + f"Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%\n"
        + f"Sequence Length Mean: {(sum(seqs_length) / len(seqs_length)):.2f}, Mediam: {statistics.median(seqs_length)}"
    )
    print(show_info)

    print("Begin extracting meta infos...")

    if data_type == "Amazon":
        meta_infos = dataset.process_meta(data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes, attributes = (
            get_attribute_Amazon(meta_infos, data_maps, attribute_core)
        )
        id2meta = meta_map(
            meta_infos,
            data_maps["item2id"],
            attributes,
            features_needed,
            attribute_core,
            prompt_format,
        )
        # item2review = get_item_review_map(review_mapping, data_maps, meta_infos)
    elif data_type == "steam":
        meta_infos = dataset.process_meta_infos()  # key is in item_id
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_steam(
            meta_infos, data_maps, attribute_core
        )
        id2meta = meta_map(
            meta_infos,
            data_maps["item2id"],
            None,
            features_needed,
            attribute_core,
            prompt_format,
        )
    elif data_type in ("Yelp", "Yelp_oracle", "Yelp_daytime"):
        meta_infos = dataset.process_meta(data_maps)
        # Yelp doesn't have brand/genre attributes like Amazon/Steam
        # Skip attribute extraction, just create id2meta mapping
        item2attributes = {}
        attribute_num = 0
        avg_attribute = 0.0
        id2meta = meta_map(
            meta_infos,
            data_maps["item2id"],
            None,  # No attributes for Yelp
            features_needed,
            attribute_core,
            prompt_format,
        )

    print(
        f"{dataset_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}"
        f"& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&"
        f"{avg_attribute:.1f} \\"
    )

    # -------------- Save Data ---------------

    with open(data_file, "w") as out:
        for (
            user,
            items,
        ) in user_items_id.items():  # user_items_id, the item is in item_id2 domain
            out.write(user + " " + " ".join(items) + "\n")

    json_str = json.dumps(id2meta)
    with open(id2meta_file, "w") as out:
        out.write(json_str)

    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, "w") as out:
        out.write(json_str)

    # For Yelp_daytime: save per-item UTC offsets so load_data can convert timestamps on-the-fly
    if data_type == "Yelp_daytime":
        offsets_file = os.path.join(processed_data_path, f"{dataset_name}_{data_type}_item_utc_offsets.json")
        if not os.path.exists(offsets_file):
            all_raw_items = set(data_maps["item2id"].keys())
            dataset.load_business_offsets(item_ids=all_raw_items)
            # Map: mapped_item_id → UTC offset
            item_offsets = {}
            for raw_id, mapped_id in data_maps["item2id"].items():
                item_offsets[mapped_id] = dataset.biz_offset.get(
                    raw_id, dataset.DEFAULT_UTC_OFFSET
                )
            with open(offsets_file, "w") as f:
                json.dump(item_offsets, f)
            print(f"Saved item UTC offsets for {len(item_offsets)} items → {offsets_file}")
        else:
            print(f"Item UTC offsets already exist: {offsets_file}")

    return data_file, id2meta_file, item2attributes_file, user_timestamps_id
