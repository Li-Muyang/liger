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
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import set_weight_decay

from .rqvae.rqvae import RQVAE


def calc_cos_sim(model, data, config):
    if len(data.shape) > 2:
        # then there are positive samples paired
        data = data[:, 0, :]
    # calculate the cos sim with prefix match

    ids = model.get_codes(data).cpu().numpy()
    # data [N, 768], ids [N, 3]
    max_item_calculate = 1000  # subsample so that memory will not blow up

    cos_sim_array = np.zeros(config["num_layers"])
    for n_prefix in range(
        1, config["num_layers"] + 1
    ):  # range(1, 4), will result in [1, 2, 3]
        unique_prefix = np.unique(ids[:, :n_prefix], axis=0)
        this_level_cos_sim_within_cluster = []
        for this_level_prefix in unique_prefix:
            mask = (ids[:, :n_prefix] == this_level_prefix).all(axis=1)  # [N,]
            this_cluster = data[mask].cpu()
            this_cluster_num = this_cluster.shape[0]
            if this_cluster_num > 1:
                indice = torch.randperm(this_cluster_num)[:max_item_calculate]
                cos_sim = F.cosine_similarity(
                    this_cluster[indice, :, None], this_cluster.t()[None, :, indice]
                )
                cos_sim_sum = torch.tril(cos_sim, diagonal=-1).sum()
                normalization_factor = (this_cluster_num - 1) * this_cluster_num / 2
                this_level_cos_sim_within_cluster.append(
                    cos_sim_sum.item() / normalization_factor
                )
        cos_sim_array[n_prefix - 1] = np.mean(this_level_cos_sim_within_cluster)
    return cos_sim_array


def train_epoch(
    writer, model, dataloader, optimizer, config, flag_eval=False, writer_keyword=None
):
    if flag_eval:
        model.eval()
    else:
        model.train()
    beta = config["beta"]

    total_loss = 0.0
    total_rec_loss = 0.0
    total_commit_loss = 0.0

    for i_batch, batch in enumerate(dataloader):
        x_batch = batch[
            0
        ]  # [batch_size, n_embed], if positive embedding is also passed in, then it is [batch_size, 2, n_embed]

        if not flag_eval:
            optimizer.zero_grad()
        with torch.set_grad_enabled(not flag_eval):
            recon_x, commitment_loss, indices = model(x_batch)
            reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction="mean")
            loss = reconstruction_mse_loss + beta * commitment_loss

        if not flag_eval:
            loss.backward()
            grad_clip = 1.0
            _grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_rec_loss += reconstruction_mse_loss.item()
        total_commit_loss += commitment_loss.item()

    keyword = ""
    if writer_keyword is not None:
        keyword = writer_keyword + "_"
    logs = {
        f"pretrain/{keyword}reconstruction_loss": total_loss / len(dataloader),
        f"pretrain/{keyword}commitment_loss": total_commit_loss / len(dataloader),
    }

    if not flag_eval:
        logs[f"pretrain/{keyword}grad_norm"] = _grad_norm

    writer.log({**logs})


def train_rqvae(model, x, device, writer, config):
    model.to(device)
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    lr = config["lr"]

    global_step = 0
    if hasattr(torch.optim, config["optimizer"]):
        optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), lr=lr)
        if "weight_decay" in optimizer.param_groups[0]:
            set_weight_decay(optimizer, config["weight_decay"])
    else:
        raise NotImplementedError(
            f"Specified Optimizer {config['optimizer']} not implemented!!"
        )

    trainset, validationset = train_test_split(x, test_size=0.05, random_state=42)
    trainset, validationset = torch.Tensor(trainset).to(device), torch.Tensor(
        validationset
    ).to(device)

    train_dataset = TensorDataset(trainset)
    val_dataset = TensorDataset(validationset)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_eval_interval = 100
    for epoch in tqdm(range(num_epochs)):
        train_epoch(
            writer,
            model,
            dataloader,
            optimizer,
            config,
            flag_eval=False,
            writer_keyword=None,
        )

        if (epoch + 1) % n_eval_interval == 0:
            train_epoch(
                writer,
                model,
                val_dataloader,
                None,
                config,
                flag_eval=True,
                writer_keyword="eval",
            )

            cos_sim_array = calc_cos_sim(model, trainset, config)
            logs = dict({})
            for i in range(config["num_layers"]):
                logs[f"pretrain/cos_sim@L{i+1}"] = cos_sim_array[i]
            cos_sim_array = calc_cos_sim(model, validationset, config)
            for i in range(config["num_layers"]):
                logs[f"pretrain/eval_cos_sim@L{i+1}"] = cos_sim_array[i]
            writer.log({**logs})

    print("Training complete.")


def train(config, device, item_embedding, id_split, id_save_location,
          context_embedding=None, context_codes_save_path=None):

    # Check if we need context codes but don't have them
    need_context = (context_embedding is not None and context_codes_save_path
                    and not os.path.exists(context_codes_save_path))

    if os.path.exists(id_save_location) and not need_context:
        return

    # If item pkl exists but context codes missing, delete pkl to force retrain
    if os.path.exists(id_save_location) and need_context:
        print("Context codes missing — deleting item codes to retrain RQ-VAE...")
        os.remove(id_save_location)

    print("Training RQ-VAE model...")
    from utils import setup_logging

    writer = setup_logging(config)
    model_config = config["dataset"]["RQ-VAE"]

    rqvae = RQVAE(
        model_config["input_dim"], model_config["hidden_dim"],
        model_config["latent_dim"], model_config["num_layers"],
        model_config["code_book_size"], model_config["dropout"],
        latent_loss_weight=model_config["beta"],
    )

    train_rqvae(
        rqvae, item_embedding[id_split["seen"] - 1], device, writer, model_config
    )
    writer.finish()

    rqvae.to(device)
    rqvae.eval()

    # Save item codes
    with torch.no_grad():
        ids = rqvae.get_codes(item_embedding).cpu().numpy()
    with open(id_save_location, "wb") as f:
        pickle.dump(ids, f)
    print(f"Item codes saved to {id_save_location}")

    # Save context codes if needed
    if need_context:
        with torch.no_grad():
            codes = rqvae.get_codes(context_embedding.to(device)).cpu().numpy()
        with open(context_codes_save_path, "wb") as f:
            pickle.dump(codes, f)
        print(f"Context codes saved to {context_codes_save_path} (shape: {codes.shape})")
