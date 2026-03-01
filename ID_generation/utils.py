"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from tqdm import trange

def encode_context_text(config, date_context, context_embedding_save_path, device):
    """Encode context text into sentence embeddings (768d). Used as input to RQ-VAE encoder."""
    if not date_context or len(date_context) == 0:
        return None
    
    if os.path.exists(context_embedding_save_path):
        context_embedding = torch.load(context_embedding_save_path, weights_only=False)
    else:
        print("Context text embedding not found, generating...")
        content_model = config["dataset"]["content_model"]
        if "sentence-t5" in content_model:
            with torch.no_grad():
                text_embedding_model = SentenceTransformer(
                    f"sentence-transformers/{content_model}", device=device
                )
                date_descp = [value for key, value in date_context.items()]
            bs = 512 if content_model == "sentence-t5-base" else 4
            with torch.no_grad():
                embeddings = text_embedding_model.encode(
                    date_descp, convert_to_numpy=True, batch_size=bs, show_progress_bar=True,
                )
        elif "bert" in content_model:
            from transformers import BertModel, BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased").to(device)
            date_descp = [value for key, value in date_context.items()]
            embedding_list = []
            for i in trange(len(date_descp) // 32 + 1):
                input_text = date_descp[i * 32 : (i + 1) * 32]
                with torch.no_grad():
                    inputs = tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                    output = model(**inputs, return_dict=True)
                    embedding_list.append(output.last_hidden_state[:, 0])
            embeddings = torch.cat(embedding_list, dim=0).cpu().numpy()
        else:
            raise NotImplementedError
        embeddings = StandardScaler().fit_transform(embeddings)
        context_embedding = torch.Tensor(embeddings).to(device)
        torch.save(context_embedding, context_embedding_save_path)
    return context_embedding


def encode_context_with_rqvae(context_text_embedding, rqvae_model_path, config, device, cache_path=None):
    """Pass context through trained RQ-VAE encoder to get aligned latent representations.
    Returns continuous latent vector (old approach: single token via context_proj)."""
    if context_text_embedding is None:
        return None

    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached RQ-VAE context encoding from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    from .rqvae.rqvae import RQVAE

    # Reconstruct RQ-VAE model from config
    model_config = config["dataset"]["RQ-VAE"]
    rqvae = RQVAE(
        model_config["input_dim"],
        model_config["hidden_dim"],
        model_config["latent_dim"],
        model_config["num_layers"],
        model_config["code_book_size"],
        model_config["dropout"],
        latent_loss_weight=model_config["beta"],
    )
    rqvae.load_state_dict(torch.load(rqvae_model_path, map_location=device, weights_only=False))
    rqvae.to(device)
    rqvae.eval()

    with torch.no_grad():
        encoded = rqvae.encode(context_text_embedding.to(device))  # [N, latent_dim]

    print(f"Context encoded via RQ-VAE encoder: {context_text_embedding.shape} → {encoded.shape}")

    if cache_path:
        torch.save(encoded, cache_path)

    return encoded


def tokenize_context_with_rqvae(context_text_embedding, rqvae_model_path, config, device, codebook_size, cache_path=None):
    """
    Tokenize context via RQ-VAE quantization (same codebook as items).
    Returns discrete semantic ID codes with expand_id offsets applied,
    so they can be looked up directly in model.shared (the T5 embedding table).

    Args:
        context_text_embedding: [N_dates, input_dim] sentence embeddings
        rqvae_model_path: path to trained RQ-VAE model
        config: full config dict
        device: torch device
        codebook_size: codebook size for expand_id offset
        cache_path: optional path to cache tokenized codes

    Returns:
        context_codes: LongTensor [N_dates, n_codebook] with expand_id offsets applied
            These are token IDs in the same vocab space as item semantic IDs.
    """
    if context_text_embedding is None:
        return None

    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached RQ-VAE context tokenization from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    from .rqvae.rqvae import RQVAE

    model_config = config["dataset"]["RQ-VAE"]
    rqvae = RQVAE(
        model_config["input_dim"],
        model_config["hidden_dim"],
        model_config["latent_dim"],
        model_config["num_layers"],
        model_config["code_book_size"],
        model_config["dropout"],
        latent_loss_weight=model_config["beta"],
    )
    rqvae.load_state_dict(torch.load(rqvae_model_path, map_location=device, weights_only=False))
    rqvae.to(device)
    rqvae.eval()

    with torch.no_grad():
        # get_codes: encode → quantize → return discrete codebook indices
        codes = rqvae.get_codes(context_text_embedding.to(device))  # [N_dates, num_levels]

    # Apply expand_id offset so codes align with T5 vocab
    # Same logic as expand_id_arr in load_data.py:
    #   codes[:, i] += codebook_size * i + 1
    n_levels = codes.shape[1]
    for i in range(n_levels):
        codes[:, i] += codebook_size * i + 1

    context_codes = codes.long().cpu()
    print(f"Context tokenized via RQ-VAE: {context_text_embedding.shape} → {context_codes.shape}")
    print(f"  Code ranges: {[f'level {i}: [{context_codes[:, i].min()}, {context_codes[:, i].max()}]' for i in range(n_levels)]}")

    if cache_path:
        torch.save(context_codes, cache_path)

    return context_codes

def process_embeddings(
    config,
    device,
    id2meta_file,
    embedding_save_path,
):
    item_2_text = json.loads(open(id2meta_file).read())
    item_id_2_text = {}
    for k, v in item_2_text.items():
        item_id_2_text[int(k)] = v

    if os.path.exists(embedding_save_path):
        item_embedding = torch.load(embedding_save_path, weights_only=False)

    else:
        print("Embeddings not found, generating embeddings...")
        content_model = config["dataset"]["content_model"]
        if "sentence-t5" in content_model:
            with torch.no_grad():
                text_embedding_model = SentenceTransformer(
                    f"sentence-transformers/{content_model}", device=device
                )
                sorted_text = [value for key, value in sorted(item_id_2_text.items())]
            bs = 512 if content_model == "sentence-t5-base" else 4
            # embedding is generated based on the sorted text
            with torch.no_grad():
                embeddings = text_embedding_model.encode(
                    sorted_text,
                    convert_to_numpy=True,
                    batch_size=bs,
                    show_progress_bar=True,
                )
            # embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # it is by default have norm 1, this is to just ensure
            with open(embedding_save_path, "wb") as f:
                pickle.dump(embeddings, f)
        elif "bert" in content_model:
            from transformers import BertModel, BertTokenizer

            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased").to(device)
            bs = 32
            # embedding is generated based on the sorted text
            sorted_text = [value for key, value in sorted(item_id_2_text.items())]
            total_length = len(sorted_text)
            embedding_list = []
            for i in trange(total_length // bs + 1):
                input_text = sorted_text[i * bs : (i + 1) * bs]
                with torch.no_grad():
                    inputs = tokenizer(
                        input_text,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(device)
                    output = model(**inputs, return_dict=True)  # [bs, 1024]
                    embeddings = output.last_hidden_state[:, 0]
                    embedding_list.append(embeddings)

            embeddings = torch.cat(embedding_list, dim=0).cpu().numpy()
            with open(embedding_save_path, "wb") as f:
                pickle.dump(embeddings, f)
        else:
            raise NotImplementedError

        embeddings = StandardScaler().fit_transform(embeddings)
        item_embedding = torch.Tensor(embeddings).to(device)
        torch.save(item_embedding, embedding_save_path)
    return item_embedding


def process_data_split(
    config,
    data_file,
    id2meta_file,
    is_steam=False,
):

    max_items_per_seq = config["dataset"]["max_items_per_seq"]

    item_2_text = json.loads(open(id2meta_file).read())
    item_id_2_text = {}
    for k, v in item_2_text.items():
        item_id_2_text[int(k)] = v

    user_sequence = []
    user_ids = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            parts = line.split()
            if not parts:
                continue
            user_ids.append(parts[0])
            user_sequence.append(
                [int(x) for x in parts[1:]]
            )  # this sequence only contains item ids

    # cut the user_sequence first before filtering out unseen val and test
    user_sequence = [
        seq if len(seq) <= max_items_per_seq else seq[-max_items_per_seq:]
        for seq in user_sequence
    ]

    if is_steam:  # subsample the data for steam
        user_sequence = user_sequence[::7]
        user_ids = user_ids[::7]

    val_items = []
    test_items = []
    all_items = []
    train_items = []
    for i in range(len(user_sequence)):
        test_items.append(user_sequence[i][-1])
        if len(user_sequence[i]) > 1:
            val_items.append(user_sequence[i][-2])
        if len(user_sequence[i]) > 2:
            train_items.extend(user_sequence[i][:-2])
        all_items.extend(user_sequence[i][:])

    test_items = np.unique(test_items)
    val_items = np.unique(val_items)
    train_items = np.unique(train_items)
    all_items = np.unique(all_items)
    item_to_ind = {
        int(key): i for i, (key, _) in enumerate(sorted(item_id_2_text.items()))
    }  # this will just be {k: k-1}

    unseen_test = np.setdiff1d(test_items, train_items)
    unseen_val = np.setdiff1d(val_items, train_items)
    train_items2 = all_items[
        ~np.isin(all_items, np.concatenate((unseen_val, unseen_test)))
    ]
    assert (train_items == train_items2).sum() == train_items.shape[
        0
    ], "Something is wrong in identifying the unique training items, causing discrepancy in the different ways of filtering the train items. Please check."
    train_inds = [item_to_ind[item] for item in train_items]
    assert (np.array(train_inds) == train_items - 1).all()

    saved_semantic_id_split = {}
    saved_semantic_id_split["unseen_val"] = unseen_val
    saved_semantic_id_split["unseen_test"] = unseen_test
    saved_semantic_id_split["seen"] = train_items

    return saved_semantic_id_split, user_sequence, user_ids
