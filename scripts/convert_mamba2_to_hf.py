import re
import json
import argparse
from os import makedirs

from mamba2_torch import Mamba2ForCausalLM, Mamba2Config

import torch
from transformers import AutoTokenizer
from safetensors.torch import save_model


def convert_ssm_config_to_hf_config(config_ssm: dict) -> Mamba2Config:
    """Convert a Mamba2Config from mamba_ssm to a Mamba2Config from here."""
    hf_config = Mamba2Config()

    # set important values from config and recalculate other resulting entries
    hf_config.hidden_size = config_ssm["d_model"]
    hf_config.intermediate_size = config_ssm["d_model"] * hf_config.expand
    hf_config.num_heads = hf_config.intermediate_size // hf_config.head_dim
    hf_config.num_hidden_layers = config_ssm["n_layer"]
    hf_config.residual_in_fp32 = config_ssm["residual_in_fp32"]
    hf_config.tie_embeddings = config_ssm["tie_embeddings"]

    # padded vocab size, mostly of 16 but 32 is also very common in different models
    vocab_size = config_ssm["vocab_size"]
    pad_vocab_size_multiple = config_ssm["pad_vocab_size_multiple"]
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size

    return hf_config


def convert_ssm_to_hf(ssm_dir, output_dir):
    original_model_params = re.search(r"(?<=mamba2-).*", ssm_dir).group(0)

    print(f"Loading original config file of mamba2-{original_model_params}")
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

    print(f"Load corresponding tokenizer")
    # kinda ugly but whatever
    mamba2_to_mamba1_parameters = {
        "130m" : "130m",
        "370m" : "370m",
        "780m" : "790m",
        "1.3b" : "1.4b",
        "2.7b" : "2.8b",
    }
    hf_tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{mamba2_to_mamba1_parameters[original_model_params]}-hf")

    save_dir = f"{output_dir}/mamba2-{original_model_params}"
    print(f"Saving hf config, model, and tokenizer to {save_dir}")
    makedirs(save_dir, exist_ok=True)
    save_model(hf_model, save_dir + "/model.safetensors", metadata={'format': 'pt'})
    hf_config.save_pretrained(save_dir)
    hf_tokenizer.save_pretrained(save_dir)

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
