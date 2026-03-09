# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import traceback

import hydra

import torch
from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.train_rqvae import train as train_sid
from ID_generation.utils import process_data_split, process_embeddings, encode_context_text
from omegaconf import DictConfig
from src.training import train_tiger
from src.load_data import load_date_context
from utils import set_seed


class set_dir:
    def __init__(self, config):
        self.directory = "./ID_generation/preprocessing/raw_data/"
        self.directory_processed = "./ID_generation/preprocessing/processed/"
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.directory_processed, exist_ok=True)

        if config["test_method"] in ["tiger", "liger"]:
            self.rqvae_save_dir = "./ID_generation/ID/"
            os.makedirs(self.rqvae_save_dir, exist_ok=True)

            id_filename = (
                f"{config['dataset']['name']}_{config['dataset']['content_model']}"
            )
            self.id_save_location = os.path.join(
                self.rqvae_save_dir, id_filename + f"_{config['seed']}.pkl"
            )

        self.embedding_save_name = f"_{config['dataset']['content_model']}"
        self.embedding_save_path = os.path.join(
            self.directory_processed, id_filename + "_embeddings.pt"
        )
        # Include type + code in context cache paths for ablation variants
        context_code = config['dataset'].get('code', '')
        context_cache_prefix = f"{id_filename}_{config['dataset']['type']}"
        if context_code:
            context_cache_prefix += f"_{context_code}"
        self.context_embedding_save_path = os.path.join(
            self.directory_processed, context_cache_prefix + "_context_embeddings.pt"
        )
        self.result_save_dir = f"./results/{config['test_method']}/"
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


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:

    # print(config)
    device = (
        torch.device(f"cuda:{config['device_id']}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    set_seed(config["seed"])

    PATH_CONFIG = set_dir(config)
    config = PATH_CONFIG.set_config(config)
    config["logging"]["project"] = "liger"
    is_steam = config["dataset"]["type"] == "steam"
    context_tokenization = config["dataset"].get("context_tokenization", None)

    try:
        data_file, id2meta_file, item2attribute_file, user_timestamps = preprocessing(config["dataset"])
        
        # Get date context file path from config, or use default
        date_context_path = config["dataset"].get(
            "date_context_file",
            f"{config['dataset']['raw_data_path']}/{config['dataset']['type']}/{config['dataset']['name']}_date_context.jsonl"
        )
        
        # Load date context if file path is provided and not disabled
        date_context, date2id = {}, None
        if date_context_path not in [None, "null", ""]:
            date_context, date2id = load_date_context(
                filepath=date_context_path,
                mapping_save_path=os.path.join(
                    config["dataset"]["processed_data_path"],
                    f"{config['dataset']['name']}_{config['dataset']['type']}_date2id.json",
                ),
            )
        else:
            print("Context loading disabled (date_context_file=null)")
        # id2meta_file: the file that save item_id to meta info, we will later use it for sentence T5 embedding generation
        # data_file: the file that save the user-item interactions.
        train_config = {
            **config["dataset"],
            **{
                k: v
                for k, v in config.items()
                if k not in ["logging", "dataset", "method"]
            },
        }
        method_config = {
            **config["method"],
            **{
                k: v
                for k, v in config.items()
                if k not in ["logging", "dataset", "method"]
            },
        }
        method_config["date_vocab_size"] = len(date2id) if date2id else 0
        # Path hint for item UTC offsets file (auto-loaded in load_data_helper for daytime conversion)
        method_config["item_offsets_path"] = os.path.join(
            config["dataset"]["processed_data_path"],
            f"{config['dataset']['name']}_{config['dataset']['type']}_item_utc_offsets.json",
        )

        # load id split
        id_split, user_sequence, user_ids = process_data_split(
            config, data_file, id2meta_file, is_steam=is_steam
        )

        # load item embedding
        item_embedding = process_embeddings(
            config, device, id2meta_file, PATH_CONFIG.embedding_save_path
        )
        # Encode context text into sentence embeddings (768d)
        context_text_embedding = None
        if date_context and len(date_context) > 0:
            context_text_embedding = encode_context_text(
                config, date_context, PATH_CONFIG.context_embedding_save_path, device=device
            )

        # Compute context codes save path (if using rqvae tokenization)
        context_codes_save_path = None
        if context_tokenization == "rqvae" and context_text_embedding is not None:
            context_codes_save_path = PATH_CONFIG.context_embedding_save_path.replace(
                "_context_embeddings.pt", "_context_codes.pkl"
            )

        train_sid(
            config, device, item_embedding, id_split, PATH_CONFIG.id_save_location,
            context_embedding=context_text_embedding if context_tokenization == "rqvae" else None,
            context_codes_save_path=context_codes_save_path,
        )

        # Load pre-computed context codes
        context_codes = None
        if context_tokenization == "rqvae" and context_codes_save_path:
            # Load pre-computed context codes (pkl, same pattern as item codes)
            import pickle
            codebook_size = config["dataset"]["RQ-VAE"]["code_book_size"]
            with open(context_codes_save_path, "rb") as f:
                raw_codes = pickle.load(f)  # numpy [N_dates, num_levels]
            # Apply expand_id offset (same as items in load_data.py)
            import torch as _torch
            context_codes = _torch.from_numpy(raw_codes).long()
            for i in range(context_codes.shape[1]):
                context_codes[:, i] += codebook_size * i + 1
            print(f"Context codes loaded: {context_codes.shape} (from {context_codes_save_path})")

        # Fail early if context is expected but codes were not produced
        assert not (method_config.get("date_vocab_size", 0) > 0 and context_codes is None), (
            "Context is enabled (date_vocab_size > 0) but context_codes could not be loaded. "
            "Ensure context_tokenization='rqvae' is set in the dataset config and that "
            "context text embeddings were successfully encoded."
        )

        method_config["context_tokenization"] = context_tokenization

        train_tiger(
            config,
            train_config,
            method_config,
            id_split,
            user_sequence,
            item_embedding,
            PATH_CONFIG.id_save_location,
            device=device,
            context_codes=context_codes,
            user_timestamps=user_timestamps,
            user_ids=user_ids,
            date2id=date2id,
        )

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise

    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
