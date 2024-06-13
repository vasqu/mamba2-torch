import json
import argparse

from .configuration_mamba2 import Mamba2Config
from .modeling_mamba2 import Mamba2ForCausalLM

import torch


def convert_ssm_config_to_hf_config(config_ssm: dict) -> Mamba2Config:
    """Convert a Mamba2Config from mamba_ssm to a Mamba2Config from transformers."""
    hf_config = Mamba2Config()

    # Set config hidden size, num hidden layers, and vocab size directly from the original config
    hf_config.hidden_size = config_ssm["d_model"]
    hf_config.intermediate_size = config_ssm["d_model"] * 2
    hf_config.num_hidden_layers = config_ssm["n_layer"]
    hf_config.residual_in_fp32 = config_ssm["residual_in_fp32"]
    hf_config.tie_embeddings = config_ssm["tie_embeddings"]

    vocab_size = config_ssm["vocab_size"]
    pad_vocab_size_multiple = config_ssm["pad_vocab_size_multiple"]
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size

    return hf_config


def convert_ssm_to_hf(ssm_dir, output_dir):
    print(f"Loading original config file")
    config_file = ssm_dir + "/config.json"
    with open(config_file, "r", encoding="utf-8") as json_file:
        original_ssm_config_dict = json.load(json_file)

    print(f"Converting to hf format and initializing empty hf model")
    hf_config = convert_ssm_config_to_hf_config(original_ssm_config_dict)
    hf_model = Mamba2ForCausalLM(hf_config)

    print(f"Load and transfer original weights to the new hf model")
    mamba_checkpoint_path = ssm_dir + "/pytorch_model.bin"
    hf_state_dict = torch.load(mamba_checkpoint_path, map_location="cpu")
    hf_model.load_state_dict(hf_state_dict)

    print(f"Saving hf config and hf model to {output_dir}")
    hf_config.save_pretrained(output_dir)
    hf_model.save_pretrained(output_dir)

    print(f"Successfully converted to hf!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_ssm_dir",
        type=str,
        required=True,
        help="Path to the directory containing the `pytorch_model.bin` mamba_ssm checkpoint file and "
             "the corresponding `config.json`.",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Path to directory to save the converted output model and config to."
    )
    args = parser.parse_args()

    convert_ssm_to_hf(args.input_ssm_dir, args.output_dir)
