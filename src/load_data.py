"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pickle

import numpy as np
import torch
from tqdm import trange

import json
from datetime import datetime

def load_date_context(
    filepath="/home/ec2-user/recsys/liger/ID_generation/preprocessing/raw_data/amazon/amazon_beauty_date_context.jsonl",
    mapping_save_path=None,
):
    date_context = {}
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            date_key = data.get("recordId")
            if not date_key:
                continue
            output = data.get("modelOutput", {}).get("content", [])
            if not output or "text" not in output[0]:
                continue
            date_context[date_key] = output[0]["text"]

    if mapping_save_path and os.path.exists(mapping_save_path):
        with open(mapping_save_path, "r") as f:
            date2id = json.load(f)
        ordered_keys = [k for k, _ in sorted(date2id.items(), key=lambda kv: kv[1])]
        ordered_context = {k: date_context[k] for k in ordered_keys if k in date_context}
    else:
        ordered_keys = sorted(date_context.keys())
        date2id = {k: i + 1 for i, k in enumerate(ordered_keys)}
        ordered_context = {k: date_context[k] for k in ordered_keys}
        if mapping_save_path:
            with open(mapping_save_path, "w") as f:
                json.dump(date2id, f)

    return ordered_context, date2id


def _timestamp_to_date_key(timestamp):
    if isinstance(timestamp, str):
        if len(timestamp) == 10 and timestamp[4] == "-" and timestamp[7] == "-":
            return timestamp
        try:
            return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return datetime.utcfromtimestamp(int(timestamp)).strftime("%Y-%m-%d")


def expand_id(id, codebook_size):
    expanded_id = list(id)
    for i_codebook_depth in range(len(expanded_id)):
        expanded_id[i_codebook_depth] += codebook_size * i_codebook_depth + 1
    return expanded_id


def expand_id_arr(id_arr, codebook_size):
    for i_codebook_depth in range(id_arr.shape[1]):
        id_arr[:, i_codebook_depth] += codebook_size * i_codebook_depth + 1


def pad_sequence(sequence, length, pad_token):
    assert (
        len(sequence) + 1 <= length
    ), "we want the max_sequence_length to include all SIDs and one extra eos token"
    return sequence + [pad_token] + [0] * (length - len(sequence) - 1)


def get_unique_semantic_ids_by_extra_position(semantic_ids, codebook_size):
    """
    In this function, we load the semantic IDs, and then assign the extra one semantic id to avoid duplication.
    :param path: where the learned semantic ID is saved
    """
    semantic_id_2_item_count = dict({})
    item_2_semantic_id, semantic_id_2_item = {}, {}

    # NOTE: the first three semantic ID range from 0 to codebook_size - 1
    # so the last one should also start from 0 to last_codebook_size - 1
    # otherwise, the max(4th semantic ID) may = user_id (in very rare case)
    for i in range(len(semantic_ids)):
        id = semantic_ids[i]  # list of length codebook_depth
        count_dict = semantic_id_2_item_count
        for i_depth in range(len(id)):
            if id[i_depth] not in count_dict.keys():
                count_dict[id[i_depth]] = dict({})
            count_dict = count_dict[id[i_depth]]
        id_dict = count_dict
        id_dict[len(id_dict)] = i + 1
        last_semantic_id = len(id_dict) - 1
        item_2_semantic_id[i + 1] = (*id, last_semantic_id)
        semantic_id_2_item[item_2_semantic_id[i + 1]] = i + 1
    # item_2_semantic_id: {item_id: [semantic_ids]}, item_id start from 1

    assert (
        len(item_2_semantic_id) == semantic_ids.shape[0]
    ), "Not all semanticid -> item collisions have been avoided!"

    # check if last dimension of semantic ids exceeds the number of available codebook entries
    all_semantic_ids_list = [
        item_2_semantic_id[idx] for idx in np.arange(1, len(semantic_ids) + 1)
    ]
    all_semantic_ids = np.array(all_semantic_ids_list)  # [n_items, n_codebook]
    max_last_semantic_ids = (
        all_semantic_ids[:, -1].max() + 1
    )  # this is actually the number of unique semantic id in the 4th position
    expand_id_arr(all_semantic_ids, codebook_size)
    return item_2_semantic_id, max_last_semantic_ids


def generate_input_sequence(
    user_id,
    user_sequence,
    item_2_semantic_id,
    max_items_per_seq,
    max_sequence_length,
    codebook_size,
    item_embedding,
    id_only,
    timestamps=None,
):
    input_sids, input_ids, attention_mask_sids, attention_mask_ids = [], [], [], []
    if user_id is not None:  # indicating that we are using user_id
        if not id_only:
            input_sids.append(user_id)
            attention_mask_sids.append(1)
        input_ids.append(user_id)
        attention_mask_ids.append(1)

    input_embeddings, labels_sids, labels_ids, label_embeddings = [], [], [], []

    for i in range(len(user_sequence)):
        if i == len(user_sequence) - 1:
            if not id_only:
                labels_sids.extend(
                    expand_id(item_2_semantic_id[user_sequence[i]], codebook_size)
                )
            this_item_embedding = item_embedding[[user_sequence[i] - 1]]
            # item id starts from 1, 1 ~ n_item
            # item_embedding shape [n_item, 768]
            labels_ids.append(user_sequence[i])
            label_embeddings = this_item_embedding  # [1, 768]
        else:
            if not id_only:
                input_semantic_ids = expand_id(
                    item_2_semantic_id[user_sequence[i]], codebook_size
                )
                input_sids.extend(input_semantic_ids)
            input_ids.append(user_sequence[i])
            attention_mask_sids.extend([1] * len(input_semantic_ids))
            attention_mask_ids.append(1)
            this_item_embedding = item_embedding[[user_sequence[i] - 1]]
            input_embeddings.append(this_item_embedding)

    if not id_only:
        labels_sids = np.array(labels_sids)
        input_sids = np.array(
            pad_sequence(input_sids, max_sequence_length, pad_token=0)
        )
        attention_mask_sids = np.array(
            pad_sequence(attention_mask_sids, max_sequence_length, pad_token=0)
        )
        assert not np.any(labels_sids == 0)
        labels_sids[labels_sids == 0] = -100

    labels_ids = np.array(labels_ids)
    input_ids = np.array(pad_sequence(input_ids, max_sequence_length, pad_token=0))
    attention_mask_ids = np.array(
        pad_sequence(attention_mask_ids, max_sequence_length, pad_token=0)
    )
    input_embeddings = torch.cat(
        input_embeddings
        + [torch.zeros_like(input_embeddings[0])]
        * (max_items_per_seq - len(input_embeddings)),
        dim=0,
    )

    # Handle timestamps
    input_timestamps = np.zeros(max_items_per_seq, dtype=np.int64)
    label_timestamp = 0
    if timestamps is not None:
        for i in range(len(user_sequence)):
            if i == len(user_sequence) - 1:
                label_timestamp = timestamps[i]
            else:
                input_timestamps[i] = timestamps[i]
        assert len(timestamps) == len(user_sequence), "Timestamps length mismatch with user_sequence"
        assert label_timestamp > 0, "Label timestamp should be positive"

    return (
        input_sids,
        input_ids,
        input_embeddings,
        attention_mask_sids,
        attention_mask_ids,
        labels_sids,
        labels_ids,
        label_embeddings,
        input_timestamps,
        label_timestamp,
    )


def load_data_helper(
    user_sequence,
    user_ids,
    unseen_val,
    unseen_test,
    item_2_semantic_id,
    item_embedding,
    method_config,
    max_items_per_seq,
    max_sequence_length,
    user_id_offset,
    codebook_size,
    id_only=False,
    user_timestamps=None,
    date2id=None,
):
    total_user_dict = {}
    for key in ["train", "val", "unseen_val", "test", "unseen_test"]:
        total_user_dict[key] = {
            "input_ids": [],
            "attention_mask_ids": [],
            "labels_ids": [],
            "input_embeddings": [],
            "label_embeddings": [],
            "input_timestamps": [],
            "label_timestamps": [],
            "label_date_ids": [],
            "input_date_ids": [],
        }
        if not id_only:
            total_user_dict[key]["input_sids"] = []
            total_user_dict[key]["labels_sids"] = []
            total_user_dict[key]["attention_mask_sids"] = []

    len_user_sequence = len(user_sequence)

    for i in trange(len_user_sequence):
        if not id_only and method_config["include_user_id"]:
            # use Hashing Trick to map the user to 2000 user tokens
            user_id = user_id_offset + i % 2000
        else:
            user_id = None

        # Get timestamps for this user if available
        user_ts = None
        user_date_ids = None
        if user_timestamps is not None:
            user_id_key = user_ids[i] if user_ids is not None else str(i + 1)
            user_id_key = str(user_id_key)
            if user_id_key in user_timestamps:
                user_ts = user_timestamps[user_id_key]
                seq_len = len(user_sequence[i])
                if len(user_ts) > seq_len:
                    user_ts = user_ts[-seq_len:]
                elif len(user_ts) < seq_len:
                    raise ValueError(
                        f"user_timestamps length mismatch for user {user_id_key}: "
                        f"{len(user_ts)} vs {seq_len}"
                    )
                if date2id is not None:
                    user_date_ids = []
                    for ts in user_ts:
                        date_key = _timestamp_to_date_key(ts)
                        if date_key not in date2id:
                            raise KeyError(
                                f"Missing date key {date_key} in date2id mapping."
                            )
                        user_date_ids.append(date2id[date_key])

        # user sequence = [1,2,3,4,5]
        # train: j = 2,3 => [1,2], [1,2,3]
        # val: j = 4 =>[1,2,3,4]
        # test: j = 5 => [1,2,3,4,5]
        for j in range(2, len(user_sequence[i]) + 1):
            this_sequence = user_sequence[i][:j]
            if j == len(user_sequence[i]) - 1:
                this_key = "unseen_val" if this_sequence[-1] in unseen_val else "val"
            elif j == len(user_sequence[i]):
                this_key = "unseen_test" if this_sequence[-1] in unseen_test else "test"
            else:
                this_key = "train"

            (
                input_sids,
                input_ids,
                input_embeddings,
                attention_mask_sids,
                attention_mask_ids,
                labels_sids,
                labels_ids,
                labels_embeddings,
                input_timestamps_seq,
                label_timestamp,
            ) = generate_input_sequence(
                user_id,
                this_sequence,
                item_2_semantic_id,
                max_items_per_seq,
                max_sequence_length,
                codebook_size,
                item_embedding,
                id_only,
                timestamps=user_ts[:j] if user_ts is not None else None,
            )
            label_date_id = (
                user_date_ids[j - 1] if user_date_ids is not None else 0
            )
            input_date_ids = np.zeros(max_items_per_seq, dtype=np.int64)
            if user_date_ids is not None:
                input_date_ids[: j - 1] = user_date_ids[: j - 1]
            if not id_only:
                total_user_dict[this_key]["input_sids"].append(input_sids)
                total_user_dict[this_key]["attention_mask_sids"].append(
                    attention_mask_sids
                )
                total_user_dict[this_key]["labels_sids"].append(labels_sids)

            total_user_dict[this_key]["input_ids"].append(input_ids)
            total_user_dict[this_key]["input_embeddings"].append(input_embeddings.cpu())
            total_user_dict[this_key]["attention_mask_ids"].append(attention_mask_ids)
            total_user_dict[this_key]["labels_ids"].append(labels_ids)
            total_user_dict[this_key]["label_embeddings"].append(
                labels_embeddings.cpu()
            )
            total_user_dict[this_key]["input_timestamps"].append(input_timestamps_seq)
            total_user_dict[this_key]["label_timestamps"].append(label_timestamp)
            total_user_dict[this_key]["label_date_ids"].append(label_date_id)
            total_user_dict[this_key]["input_date_ids"].append(input_date_ids)

    for key in total_user_dict.keys():
        for sub_key in total_user_dict[key].keys():
            if sub_key in ["label_embeddings"]:
                if len(total_user_dict[key][sub_key]) > 0:
                    total_user_dict[key][sub_key] = torch.cat(
                        total_user_dict[key][sub_key]
                    )
                else:
                    total_user_dict[key][sub_key] = torch.tensor([])
            elif sub_key in ["input_embeddings"]:
                if len(total_user_dict[key][sub_key]) > 0:
                    total_user_dict[key][sub_key] = torch.stack(
                        total_user_dict[key][sub_key]
                    )
                else:
                    total_user_dict[key][sub_key] = torch.tensor([])
            elif sub_key == "input_timestamps":
                if len(total_user_dict[key][sub_key]) > 0:
                    total_user_dict[key][sub_key] = torch.tensor(
                        np.stack(total_user_dict[key][sub_key]), dtype=torch.long
                    )
                else:
                    total_user_dict[key][sub_key] = torch.tensor([], dtype=torch.long)
            else:
                total_user_dict[key][sub_key] = torch.tensor(
                    total_user_dict[key][sub_key], dtype=torch.long
                )

    return (
        total_user_dict["train"],
        total_user_dict["val"],
        total_user_dict["unseen_val"],
        total_user_dict["test"],
        total_user_dict["unseen_test"],
    )


def load_data(
    path,
    user_sequence,
    user_ids,
    unseen_val,
    unseen_test,
    seen,
    item_embedding,
    method_config,
    max_length=258,
    codebook_size=256,
    max_items_per_seq=np.inf,
    user_timestamps=None,
    date2id=None,
):
    """
    :param path: path to load the semantic ID
    :param user_sequence: user sequence
    :param unseen_val, unseen_test, seen: semantic ID np.array
    :param item_embedding: [n_item, n_embd], where n_embd is sentence-T5 embedding dimension
    :param max_length: for the generated sequence
    :param max_items_per_seq: number of items in each sequence
    """
    n_semantic_codebook = 3

    semantic_ids = pickle.load(open(path, "rb"))

    item_2_semantic_id, max_last_semantic_ids = (
        get_unique_semantic_ids_by_extra_position(semantic_ids, codebook_size)
    )
    last_codebook_size = max(max_last_semantic_ids, codebook_size)
    n_codebook = n_semantic_codebook + 1  # actual number of codebook

    # split into train/val/test items
    val_unseen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in unseen_val])
    expand_id_arr(val_unseen_semantic_ids, codebook_size)
    test_unseen_semantic_ids = np.array(
        [item_2_semantic_id[idx] for idx in unseen_test]
    )
    expand_id_arr(test_unseen_semantic_ids, codebook_size)
    seen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in seen])
    expand_id_arr(seen_semantic_ids, codebook_size)
    all_semantic_ids = np.array(
        [item_2_semantic_id[idx] for idx in item_2_semantic_id.keys()]
    )
    expand_id_arr(all_semantic_ids, codebook_size)

    # get semantic id - embedding dict
    semantic_id_2_embd = dict({})
    for key in item_2_semantic_id.keys():  # again, the key here start from 1
        semantic_id_2_embd[tuple(expand_id(item_2_semantic_id[key], codebook_size))] = (
            item_embedding[key - 1]
        )

    user_id_offset = 1 + n_semantic_codebook * codebook_size + last_codebook_size
    training_data, val_data, unseen_val_data, test_data, unseen_test_data = (
        load_data_helper(
            user_sequence,
            user_ids,
            unseen_val,
            unseen_test,
            item_2_semantic_id,
            item_embedding,
            method_config,
            max_items_per_seq,
            max_length,
            user_id_offset,
            codebook_size,
            user_timestamps=user_timestamps,
            date2id=date2id,
        )
    )

    return (
        training_data,
        val_data,
        test_data,
        unseen_val_data,
        unseen_test_data,
        seen_semantic_ids,
        val_unseen_semantic_ids,
        test_unseen_semantic_ids,
        max_last_semantic_ids,
        n_semantic_codebook,
        n_codebook,
        all_semantic_ids,
    )


def load_data_id(
    user_sequence,
    user_ids,
    unseen_val,
    unseen_test,
    item_embedding,
    method_config,
    max_length=258,
    max_items_per_seq=np.inf,
    user_timestamps=None,
    date2id=None,
):

    training_data, val_data, unseen_val_data, test_data, unseen_test_data = (
        load_data_helper(
            user_sequence,
            user_ids,
            unseen_val,
            unseen_test,
            None,
            item_embedding,
            method_config,
            max_items_per_seq,
            max_length,
            None,
            None,
            id_only=True,
            user_timestamps=user_timestamps,
            date2id=date2id,
        )
    )

    return training_data, val_data, test_data, unseen_val_data, unseen_test_data