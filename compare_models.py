"""
Comparative analysis script for TIGER models with and without context (date) information.

Model A: TIGER trained with context (date) information  — "liger" variant
Model B: TIGER trained without context information      — baseline variant

Usage (run from dev-branches/v4.2/liger/):
    python compare_models.py \
        dataset=yelp \
        dataset.name=Yelp \
        seed=42 \
        device_id=0 \
        +ckpt_a=/path/to/context_model/results/ckpt_best.pt \
        +ckpt_b=/path/to/no_context_model/results/ckpt_best.pt \
        +output_csv=./results/compare_results.csv \
        +top_k=10

Notes:
  - The preprocessing pipeline (data_file, id_save_location, embeddings, date2id mapping)
    is executed exactly as in run.py / train_tiger(), guaranteeing identical test splits.
  - Both models are evaluated on the same DataLoader(shuffle=False) instances.
  - Model B simply has date_vocab_size=0 so _get_context_embeds() returns None.
  - All heavy lifting (data loading, model construction, inference, metric computation)
    reuses existing functions from the repo without modification.
"""

import csv
import os
import pickle
import sys
import traceback
from collections import defaultdict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import T5Config

# ── Existing repo imports ────────────────────────────────────────────────────
from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.train_rqvae import train as train_sid
from ID_generation.utils import (
    encode_context_text,
    process_data_split,
    process_embeddings,
)
from src.evaluation import calculate_metrics, evaluate, model_forward
from src.load_data import load_data, load_date_context
from src.tiger import TIGER
from utils import CustomDataset, set_seed

import json as _json  # used for loading id2meta


# ─────────────────────────────────────────────────────────────────────────────
# Path configuration (identical to run.py set_dir)
# ─────────────────────────────────────────────────────────────────────────────

class set_dir:
    def __init__(self, config):
        self.directory = "./ID_generation/preprocessing/raw_data/"
        self.directory_processed = "./ID_generation/preprocessing/processed/"
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.directory_processed, exist_ok=True)

        self.rqvae_save_dir = "./ID_generation/ID/"
        os.makedirs(self.rqvae_save_dir, exist_ok=True)

        id_filename = (
            f"{config['dataset']['name']}_{config['dataset']['content_model']}"
        )
        self.id_save_location = os.path.join(
            self.rqvae_save_dir, id_filename + f"_{config['seed']}.pkl"
        )
        self.embedding_save_path = os.path.join(
            self.directory_processed, id_filename + "_embeddings.pt"
        )
        self.context_embedding_save_path = os.path.join(
            self.directory_processed, id_filename + "_context_embeddings.pt"
        )
        self.result_save_dir = f"./results/compare/"
        os.makedirs(self.result_save_dir, exist_ok=True)

    def set_config(self, config):
        config["dataset"]["raw_data_path"] = self.directory
        config["dataset"]["processed_data_path"] = self.directory_processed
        config["output_path"] = os.path.join(
            self.result_save_dir,
            f"{config['dataset']['type']}_{config['dataset']['name']}",
            f"{config['experiment_id']}_seed_{config['seed']}",
        )
        os.makedirs(config["output_path"], exist_ok=True)
        return config


# ─────────────────────────────────────────────────────────────────────────────
# Model builder — mirrors the init block inside train_tiger()
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    config,           # dataset-level config (TIGER sub-dict)
    method_config,
    codebook_size,
    max_items_per_seq,
    n_semantic_codebook,
    max_last_semantic_ids,
    item_embedding,
    device,
    ckpt_path,
    context_codes=None,
    with_context=True,
):
    """
    Construct a TIGER model and load a checkpoint.
    Mirrors the model-construction logic in training.py::train_tiger().
    """
    last_codebook_size = max(max_last_semantic_ids, codebook_size)

    if method_config["include_user_id"]:
        this_vocab_size = (
            2000 + codebook_size * n_semantic_codebook + last_codebook_size + 2
        )
    else:
        this_vocab_size = codebook_size * n_semantic_codebook + last_codebook_size + 2

    if method_config["use_id"] == "item_id":
        this_vocab_size = item_embedding.shape[0] + 2

    t5_config = config["T5"]
    effective_n_positions = config["n_positions"]

    model_config = T5Config(
        num_layers=t5_config["encoder_layers"],
        num_decoder_layers=t5_config["decoder_layers"],
        d_model=t5_config["d_model"],
        d_ff=t5_config["d_ff"],
        num_heads=t5_config["num_heads"],
        d_kv=t5_config["d_kv"],
        dropout_rate=t5_config["dropout_rate"],
        vocab_size=this_vocab_size,
        pad_token_id=0,
        eos_token_id=int(this_vocab_size - 1),
        decoder_start_token_id=0,
        feed_forward_proj=t5_config["feed_forward_proj"],
        n_positions=effective_n_positions,
        layer_norm_epsilon=1e-8,
        initializer_factor=t5_config["initializer_factor"],
    )

    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
    ).to(device)

    # ── Attach RQ-VAE tokenized context ──────────────────────────────────────
    if with_context and context_codes is not None:
        model.context_codes = context_codes.to(device)
        print(f"[Model A] Using RQ-VAE tokenized context: {context_codes.shape}")
    else:
        model.context_codes = None
        if not with_context:
            print("[Model B] Context disabled")
        else:
            print("[Model A] No context codes available")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Support both raw state_dict and training_state checkpoint formats
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Per-example inference — thin wrapper around the existing evaluate()
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, dataloader, all_semantic_ids, device, method_config, top_k):
    """
    Calls the existing evaluate() function and returns aligned per-example lists.

    Returns:
        recall_lists  : dict[k] -> List[float]  (0.0 or 1.0 per example)
        ndcg_lists    : dict[k] -> List[float]
        cand_list     : List[Tensor[num_return_seq, n_codebook]]
        date_id_list  : List[int]  (label_date_id per example, 0 if unavailable)
    """
    KEYS = [top_k]
    RETRIEVE_KEY = [top_k]

    recall_dict, ndcg_dict, returned_cand, _ = evaluate(
        model,
        dataloader,
        all_semantic_ids,
        device,
        method_config=method_config,
        KEYS=KEYS,
        RETRIEVE_KEY=RETRIEVE_KEY,
    )

    # Collect label_date_ids in dataloader order (shuffle=False guaranteed)
    date_id_list = []
    for batch in dataloader:
        if "label_date_ids" in batch:
            date_id_list.extend(batch["label_date_ids"].tolist())
        else:
            date_id_list.extend([0] * len(batch["labels_sids"]))

    return recall_dict, ndcg_dict, returned_cand, date_id_list


# ─────────────────────────────────────────────────────────────────────────────
# Comparative analysis
# ─────────────────────────────────────────────────────────────────────────────

def compare_per_example(
    recall_a, ndcg_a, cand_a,
    recall_b, ndcg_b, cand_b,
    date_id_list,
    split_name,
    top_k,
    example_offset=0,
):
    """
    Zip per-example results from Model A and Model B into comparison rows.

    Returns a list of dicts, one per example.
    """
    n = len(recall_a[top_k])
    rows = []
    for i in range(n):
        ra = float(recall_a[top_k][i])
        rb = float(recall_b[top_k][i])
        na = float(ndcg_a[top_k][i])
        nb = float(ndcg_b[top_k][i])

        if ra == 1.0 and rb == 0.0:
            winner = "A_only"
        elif ra == 0.0 and rb == 1.0:
            winner = "B_only"
        elif ra == 1.0 and rb == 1.0:
            winner = "both"
        else:
            winner = "neither"

        # Serialize top predicted SIDs as readable strings
        top_sids_a = cand_a[i][0].tolist() if len(cand_a[i]) > 0 else []
        top_sids_b = cand_b[i][0].tolist() if len(cand_b[i]) > 0 else []

        rows.append({
            "example_idx": example_offset + i,
            "split_type": split_name,
            "label_date_id": date_id_list[i],
            f"recall_a@{top_k}": ra,
            f"recall_b@{top_k}": rb,
            f"ndcg_a@{top_k}": round(na, 6),
            f"ndcg_b@{top_k}": round(nb, 6),
            "winner": winner,
            "ndcg_delta": round(na - nb, 6),  # positive = A better
            "top_pred_sids_a": str(top_sids_a),
            "top_pred_sids_b": str(top_sids_b),
        })
    return rows


def print_summary(all_rows, top_k):
    """Print a console summary table of the comparative analysis."""
    splits = sorted(set(r["split_type"] for r in all_rows))
    buckets = ["A_only", "B_only", "both", "neither"]

    print("\n" + "=" * 72)
    print(f"COMPARATIVE ANALYSIS SUMMARY  (Recall@{top_k} / NDCG@{top_k})")
    print("=" * 72)

    for split in ["ALL"] + splits:
        if split == "ALL":
            rows = all_rows
        else:
            rows = [r for r in all_rows if r["split_type"] == split]

        n = len(rows)
        if n == 0:
            continue

        print(f"\n--- Split: {split}  (N={n}) ---")
        print(f"  {'Bucket':<12} {'Count':>8} {'%':>8}  {'Mean NDCG-A':>12}  {'Mean NDCG-B':>12}  {'Mean Δ':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")

        for bucket in buckets:
            sub = [r for r in rows if r["winner"] == bucket]
            cnt = len(sub)
            pct = 100.0 * cnt / n if n > 0 else 0.0
            mean_na = np.mean([r[f"ndcg_a@{top_k}"] for r in sub]) if sub else float("nan")
            mean_nb = np.mean([r[f"ndcg_b@{top_k}"] for r in sub]) if sub else float("nan")
            mean_d  = np.mean([r["ndcg_delta"] for r in sub]) if sub else float("nan")
            print(f"  {bucket:<12} {cnt:>8} {pct:>7.1f}%  {mean_na:>12.4f}  {mean_nb:>12.4f}  {mean_d:>10.4f}")

        # Overall means
        mean_na_all = np.mean([r[f"ndcg_a@{top_k}"] for r in rows])
        mean_nb_all = np.mean([r[f"ndcg_b@{top_k}"] for r in rows])
        rec_a_all   = np.mean([r[f"recall_a@{top_k}"] for r in rows])
        rec_b_all   = np.mean([r[f"recall_b@{top_k}"] for r in rows])
        print(f"\n  Overall Recall@{top_k}  — A: {rec_a_all:.4f}   B: {rec_b_all:.4f}")
        print(f"  Overall NDCG@{top_k}    — A: {mean_na_all:.4f}   B: {mean_nb_all:.4f}")

    print("\n" + "=" * 72)

    # Date-level breakdown (if date_ids are available)
    if any(r["label_date_id"] != 0 for r in all_rows):
        print("\n--- NDCG delta by label_date_id (top 15 most common dates) ---")
        date_groups = defaultdict(list)
        for r in all_rows:
            date_groups[r["label_date_id"]].append(r["ndcg_delta"])
        sorted_dates = sorted(date_groups.items(), key=lambda x: -len(x[1]))[:15]
        print(f"  {'date_id':>8}  {'N':>6}  {'Mean Δ (A-B)':>14}")
        print(f"  {'-'*8}  {'-'*6}  {'-'*14}")
        for date_id, deltas in sorted_dates:
            print(f"  {date_id:>8}  {len(deltas):>6}  {np.mean(deltas):>14.4f}")
        print()


def save_csv(all_rows, output_csv, top_k):
    """Save per-example comparison rows to a CSV file."""
    if not all_rows:
        print("No rows to save.")
        return
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPer-example results saved to: {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Qualitative example printing
# ─────────────────────────────────────────────────────────────────────────────

def _sid_tuple_to_name(sid_tuple, sid2itemid, id2meta):
    """Resolve a SID tuple → item name string (with fallback)."""
    key = tuple(int(x) for x in sid_tuple)
    item_id = sid2itemid.get(key, None)
    if item_id is None:
        return f"[unknown SID {list(key)}]"
    name = id2meta.get(str(item_id), id2meta.get(item_id, None))
    if name is None:
        return f"[item_id={item_id}]"
    # Truncate to first 80 chars for readability
    return name[:80].strip()


def _item_id_to_name(item_id, id2meta):
    """Resolve an integer item_id → item name string."""
    name = id2meta.get(str(item_id), id2meta.get(item_id, None))
    if name is None:
        return f"[item_id={item_id}]"
    return name[:80].strip()


def print_qualitative_examples(
    all_rows,
    split_info,       # list of (split_name, data_dict, cand_a_list, cand_b_list, start_idx)
    id2meta,          # {str(item_id) | int(item_id) → text}
    sid2itemid,       # {tuple(expanded_sid) → item_id}
    date_context,     # {date_key_str → context_text}
    id2datekey,       # {int_date_id → date_key_str}
    top_k,
    bucket="A_only",
    n_print=5,
):
    """
    Print human-readable qualitative examples for a given bucket.

    For each selected example prints:
      - Date context seen by Model A
      - Input sequence (history) with item names
      - Ground-truth item
      - Model A top-K predictions (names)
      - Model B top-K predictions (names)
    """
    target_rows = [r for r in all_rows if r["winner"] == bucket]
    if not target_rows:
        print(f"\n[Qualitative] No examples found for bucket '{bucket}'.")
        return

    # Build a fast lookup: global example_idx → (split_name, local_idx_within_split)
    # split_info contains the ordered splits with their start indices
    def _resolve_local(global_idx):
        for (sname, data_dict, cand_a_list, cand_b_list, start) in split_info:
            end = start + len(cand_a_list)
            if start <= global_idx < end:
                return sname, data_dict, cand_a_list, cand_b_list, global_idx - start
        return None, None, None, None, None

    print("\n" + "=" * 72)
    print(f"QUALITATIVE EXAMPLES — bucket: {bucket}  (showing up to {n_print})")
    print("=" * 72)

    printed = 0
    for row in target_rows:
        if printed >= n_print:
            break

        global_idx = row["example_idx"]
        sname, data_dict, cand_a_list, cand_b_list, local_idx = _resolve_local(global_idx)
        if sname is None:
            continue

        # ── Date context ────────────────────────────────────────────────────
        label_date_id = row["label_date_id"]
        date_key = id2datekey.get(label_date_id, None) if id2datekey else None
        context_text = (
            date_context.get(date_key, "[no context]")
            if date_context and date_key
            else "[no context]"
        )

        # ── Input sequence (history) ────────────────────────────────────────
        # input_ids shape: [n_examples, max_len], integer item IDs, zero-padded
        input_ids_row = data_dict["input_ids"][local_idx]  # Tensor[max_len]
        # Strip padding zeros and the sequence is item_ids (1-indexed)
        input_item_ids = [int(x) for x in input_ids_row.tolist() if int(x) > 0]
        # If include_user_id, the first token is user_id (not item) — skip it
        # We detect this heuristically: user_id tokens are > n_items in value
        # But since we don't have n_items easily here, just show all non-zero IDs
        # The id2meta lookup will return "[item_id=X]" for user_id tokens (not in meta)
        history_names = [_item_id_to_name(iid, id2meta) for iid in input_item_ids]
        # Filter out tokens not found in id2meta (likely user_id tokens)
        history_names = [n for n in history_names if not n.startswith("[item_id=")]

        # ── Ground-truth item ───────────────────────────────────────────────
        labels_ids_row = data_dict["labels_ids"][local_idx]  # Tensor[1] or scalar
        gt_item_id = int(labels_ids_row.reshape(-1)[0])
        gt_name = _item_id_to_name(gt_item_id, id2meta)

        # ── Model A predictions (top-K SIDs → names) ────────────────────────
        # cand_a_list[local_idx]: Tensor[num_return_seq, n_codebook]
        cand_a_tensor = cand_a_list[local_idx]  # [num_return_seq, n_codebook]
        preds_a = []          # list of (resolved_item_id, name)
        for seq_idx in range(min(top_k, cand_a_tensor.shape[0])):
            sid_tuple = cand_a_tensor[seq_idx].tolist()
            key = tuple(int(x) for x in sid_tuple)
            pred_item_id = sid2itemid.get(key, None)
            preds_a.append((pred_item_id, _sid_tuple_to_name(sid_tuple, sid2itemid, id2meta)))

        # ── Model B predictions (top-K SIDs → names) ────────────────────────
        cand_b_tensor = cand_b_list[local_idx]
        preds_b = []          # list of (resolved_item_id, name)
        for seq_idx in range(min(top_k, cand_b_tensor.shape[0])):
            sid_tuple = cand_b_tensor[seq_idx].tolist()
            key = tuple(int(x) for x in sid_tuple)
            pred_item_id = sid2itemid.get(key, None)
            preds_b.append((pred_item_id, _sid_tuple_to_name(sid_tuple, sid2itemid, id2meta)))

        # ── Print ────────────────────────────────────────────────────────────
        print(f"\n[Example #{printed+1}]  global_idx={global_idx}  split={sname}  "
              f"ndcg_delta={row['ndcg_delta']:+.4f}")
        print(f"  Context (date_id={label_date_id}, date={date_key}):")
        print(f"    {context_text}")
        print(f"  Input history ({len(history_names)} items):")
        for hi, hname in enumerate(history_names):
            print(f"    [{hi+1}] {hname}")
        print(f"  Ground-truth item (item_id={gt_item_id}):")
        print(f"    → {gt_name}")
        print(f"  Model A (context-aware) top-{top_k} predictions:")
        for pi, (pred_id_a, pname_a) in enumerate(preds_a):
            # ✓ marks whichever rank position matches the ground-truth item (recall)
            hit = "✓" if pred_id_a == gt_item_id else " "
            print(f"    {hit}[{pi+1:2d}] {pname_a}")
        print(f"  Model B (no context)   top-{top_k} predictions:")
        for pi, (pred_id_b, pname_b) in enumerate(preds_b):
            hit = "✓" if pred_id_b == gt_item_id else " "
            print(f"    {hit}[{pi+1:2d}] {pname_b}")

        printed += 1

    print(f"\n{'='*72}")
    print(f"Showed {printed}/{len(target_rows)} '{bucket}' examples.")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:

    # ── Extra CLI params (not in base configs) ────────────────────────────────
    ckpt_a     = config.get("ckpt_a", None)
    ckpt_b     = config.get("ckpt_b", None)
    output_csv = config.get("output_csv", "./results/compare/compare_results.csv")
    top_k      = int(config.get("top_k", 10))

    if ckpt_a is None or ckpt_b is None:
        raise ValueError(
            "Both +ckpt_a and +ckpt_b must be provided.\n"
            "Example:\n"
            "  python compare_models.py +ckpt_a=/path/a/ckpt_best.pt "
            "+ckpt_b=/path/b/ckpt_best.pt"
        )

    # Hydra places +date_context_file at the top-level config, but the pipeline
    # reads config["dataset"]["date_context_file"].  Inject the override here
    # before anything else runs so load_date_context() picks it up correctly.
    date_context_file_override = config.get("date_context_file", None)
    if date_context_file_override is not None:
        config["dataset"]["date_context_file"] = date_context_file_override
        print(f"[Override] date_context_file → {date_context_file_override}")

    device = (
        torch.device(f"cuda:{config['device_id']}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    set_seed(config["seed"])

    # ── Path setup (identical to run.py) ─────────────────────────────────────
    PATH_CONFIG = set_dir(config)
    config = PATH_CONFIG.set_config(config)

    is_steam = config["dataset"]["type"] == "steam"
    context_tokenization = config["dataset"].get("context_tokenization", None)

    try:
        # ── Step 1: Preprocessing (identical to run.py main()) ────────────────
        data_file, id2meta_file, item2attribute_file, user_timestamps = preprocessing(
            config["dataset"]
        )

        date_context_path = config["dataset"].get(
            "date_context_file",
            f"{config['dataset']['raw_data_path']}/{config['dataset']['type']}"
            f"/{config['dataset']['name']}_date_context.jsonl",
        )

        date_context, date2id = {}, None
        if date_context_path not in [None, "null", ""]:
            date_context, date2id = load_date_context(
                filepath=date_context_path,
                mapping_save_path=os.path.join(
                    config["dataset"]["processed_data_path"],
                    f"{config['dataset']['name']}_date2id.json",
                ),
            )
        else:
            print("Context loading disabled (date_context_file=null)")

        train_config = {
            **config["dataset"],
            **{k: v for k, v in config.items() if k not in ["logging", "dataset", "method"]},
        }
        method_config = {
            **config["method"],
            **{k: v for k, v in config.items() if k not in ["logging", "dataset", "method"]},
        }
        method_config["date_vocab_size"] = len(date2id) if date2id else 0

        # ── Load id2meta for qualitative output (item_id → description text) ──
        # id2meta_file is the JSON saved by preprocessing: {str(item_id) → text}
        with open(id2meta_file, "r") as _f:
            id2meta = _json.load(_f)

        # ── Step 2: Data splits and embeddings ────────────────────────────────
        id_split, user_sequence, user_ids = process_data_split(
            config, data_file, id2meta_file, is_steam=is_steam
        )
        item_embedding = process_embeddings(
            config, device, id2meta_file, PATH_CONFIG.embedding_save_path
        )

        context_text_embedding = None
        if date_context and len(date_context) > 0:
            context_text_embedding = encode_context_text(
                config, date_context, PATH_CONFIG.context_embedding_save_path, device=device
            )

        # ── Step 3: RQ-VAE IDs (loads from cache if available) ───────────────
        context_codes_save_path = None
        if context_tokenization == "rqvae" and context_text_embedding is not None:
            context_codes_save_path = PATH_CONFIG.context_embedding_save_path.replace(
                "_context_embeddings.pt", "_context_codes.pkl"
            )

        train_sid(
            config, device, item_embedding, id_split, PATH_CONFIG.id_save_location,
            context_embedding=(
                context_text_embedding if context_tokenization == "rqvae" else None
            ),
            context_codes_save_path=context_codes_save_path,
        )

        # ── Step 4: Build context_codes for Model A ───────────────────────────
        context_codes = None
        if context_tokenization == "rqvae" and context_codes_save_path:
            codebook_size = config["dataset"]["RQ-VAE"]["code_book_size"]
            with open(context_codes_save_path, "rb") as f:
                raw_codes = pickle.load(f)
            context_codes = torch.from_numpy(raw_codes).long()
            for i in range(context_codes.shape[1]):
                context_codes[:, i] += codebook_size * i + 1
            print(f"Context codes loaded: {context_codes.shape}")

        # Fail early if context is expected but codes were not produced
        if method_config.get("date_vocab_size", 0) > 0 and context_codes is None:
            assert False, (
                "Context is enabled (date_vocab_size > 0) but context_codes could not be loaded. "
                "Ensure context_tokenization='rqvae' is set in the dataset config and that "
                "context text embeddings were successfully encoded."
            )

        method_config["date_token_offset"] = None
        method_config["context_tokenization"] = context_tokenization

        # ── Step 5: Load data — IDENTICAL to train_tiger() call ──────────────
        # This guarantees the same test split as was used during training.
        codebook_size = config["dataset"]["RQ-VAE"]["code_book_size"]
        max_items_per_seq = config["dataset"]["max_items_per_seq"]
        tiger_config = config["dataset"]["TIGER"]
        effective_n_positions = tiger_config["n_positions"]
        eval_batch_size = tiger_config["trainer"]["eval_batch_size"]

        unseen_val  = id_split["unseen_val"]
        unseen_test = id_split["unseen_test"]
        seen        = id_split["seen"]

        result = load_data(
            PATH_CONFIG.id_save_location,
            user_sequence,
            user_ids,
            unseen_val,
            unseen_test,
            seen,
            item_embedding,
            method_config,
            max_length=effective_n_positions,
            codebook_size=codebook_size,
            max_items_per_seq=max_items_per_seq,
            user_timestamps=user_timestamps,
            date2id=date2id,
            ooc_config=None,  # no OOC filtering for comparison
        )

        (
            _training_data,
            _val_data,
            test_data,
            _unseen_val_data,
            unseen_test_data,
            seen_semantic_ids,
            val_unseen_semantic_ids,
            test_unseen_semantic_ids,
            max_last_semantic_ids,
            n_semantic_codebook,
            n_codebook,
            item2sid,
        ) = result

        # Build all_semantic_ids (used by evaluate() for beam-search constraint)
        all_semantic_ids = np.unique(
            np.concatenate(
                [seen_semantic_ids, val_unseen_semantic_ids, test_unseen_semantic_ids],
                axis=0,
            ),
            axis=0,
        )
        all_semantic_ids = torch.from_numpy(all_semantic_ids)

        if method_config["flag_use_output_embedding"]:
            item_embedding = item_embedding.to(device)

        # ── Step 6: Build DataLoaders — shuffle=False for alignment ──────────
        test_dataset        = CustomDataset(test_data)
        unseen_test_dataset = CustomDataset(unseen_test_data)

        test_dataloader        = DataLoader(test_dataset,        batch_size=eval_batch_size, shuffle=False)
        unseen_test_dataloader = DataLoader(unseen_test_dataset, batch_size=eval_batch_size, shuffle=False)

        print(f"\nTest set sizes — in_set: {len(test_dataset)}  cold_start: {len(unseen_test_dataset)}")

        # ── Step 7: Build Model A (with context) ─────────────────────────────
        print("\n>>> Building Model A (with context)...")
        model_a = build_model(
            config=tiger_config,
            method_config=method_config,
            codebook_size=codebook_size,
            max_items_per_seq=max_items_per_seq,
            n_semantic_codebook=n_semantic_codebook,
            max_last_semantic_ids=max_last_semantic_ids,
            item_embedding=item_embedding,
            device=device,
            ckpt_path=ckpt_a,
            context_codes=context_codes,
            with_context=True,
        )

        # ── Step 8: Build Model B (no context) ───────────────────────────────
        # Make a copy of method_config with date_vocab_size=0 for Model B so
        # _get_context_embeds() in evaluation.py returns None cleanly.
        method_config_b = dict(method_config)
        method_config_b["date_vocab_size"] = 0

        print("\n>>> Building Model B (without context)...")
        model_b = build_model(
            config=tiger_config,
            method_config=method_config_b,
            codebook_size=codebook_size,
            max_items_per_seq=max_items_per_seq,
            n_semantic_codebook=n_semantic_codebook,
            max_last_semantic_ids=max_last_semantic_ids,
            item_embedding=item_embedding,
            device=device,
            ckpt_path=ckpt_b,
            context_codes=None,
            with_context=False,
        )

        # ── Build lookup: expanded SID tuple → item_id (1-indexed) ───────────
        # item2sid shape: [n_items, n_codebook], already expand_id_arr applied
        sid2itemid = {
            tuple(int(x) for x in item2sid[i]): i + 1
            for i in range(item2sid.shape[0])
        }
        # Build reverse date mapping: int_id → date_key_str
        id2datekey = {v: k for k, v in date2id.items()} if date2id else {}

        # ── Step 9: Inference on test splits ─────────────────────────────────
        all_rows = []
        split_info = []   # for qualitative printing: (name, data_dict, cand_a, cand_b, start)
        example_offset = 0

        for split_name, dataloader, data_dict in [
            ("in_set",     test_dataloader,         test_data),
            ("cold_start", unseen_test_dataloader,  unseen_test_data),
        ]:
            n_examples = len(dataloader.dataset)
            if n_examples == 0:
                print(f"\nSkipping empty split: {split_name}")
                continue

            print(f"\n{'='*60}")
            print(f"Evaluating split: {split_name}  (N={n_examples})")
            print(f"{'='*60}")

            # ── Model A inference ─────────────────────────────────────────────
            print("  Running Model A (with context)...")
            recall_a, ndcg_a, cand_a, date_ids_a = run_inference(
                model_a, dataloader, all_semantic_ids, device, method_config, top_k
            )

            # ── Model B inference ─────────────────────────────────────────────
            # Re-iterate the same dataloader; shuffle=False guarantees same order.
            print("  Running Model B (without context)...")
            recall_b, ndcg_b, cand_b, date_ids_b = run_inference(
                model_b, dataloader, all_semantic_ids, device, method_config_b, top_k
            )

            # Sanity check: date_ids must match (same dataloader, same order)
            assert date_ids_a == date_ids_b, (
                "date_id lists differ between Model A and B — dataloader ordering mismatch!"
            )

            # ── Build per-example rows ────────────────────────────────────────
            rows = compare_per_example(
                recall_a, ndcg_a, cand_a,
                recall_b, ndcg_b, cand_b,
                date_ids_a,
                split_name,
                top_k,
                example_offset=example_offset,
            )
            all_rows.extend(rows)

            # Store split info for qualitative printing (data_dict kept in memory)
            split_info.append((split_name, data_dict, cand_a, cand_b, example_offset))
            example_offset += len(rows)

        # ── Step 10: Quantitative output ──────────────────────────────────────
        print_summary(all_rows, top_k)
        save_csv(all_rows, output_csv, top_k)

        # ── Step 11: Qualitative examples ─────────────────────────────────────
        # Print examples where Model A (context-aware) succeeds and B fails
        print_qualitative_examples(
            all_rows,
            split_info,
            id2meta=id2meta,
            sid2itemid=sid2itemid,
            date_context=date_context,
            id2datekey=id2datekey,
            top_k=top_k,
            bucket="A_only",
            n_print=5,
        )
        # Print examples where Model B succeeds and A fails (reverse analysis)
        print_qualitative_examples(
            all_rows,
            split_info,
            id2meta=id2meta,
            sid2itemid=sid2itemid,
            date_context=date_context,
            id2datekey=id2datekey,
            top_k=top_k,
            bucket="B_only",
            n_print=5,
        )

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()