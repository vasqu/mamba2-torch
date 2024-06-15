# HuggingFace Compatible Mamba2


## Introduction

 This is a highly experimental implementation of `Mamba2` [[1]](#work-this-is-based-on) that is compatible with the `transformers` library by `Hugging Face` [[2]](#work-this-is-based-on). It is only supporting the pure Mamba2 block which means the hybrid variants with Attention and/or MLP are not available.  

NOTE: You can use this repo to use `Mamba2` based models with all optimisation paths:
- Triton kernels and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) ("fastest")
- Triton kernels only (default) 
- Pure PyTorch

NOTE: I'm not affiliated with the original authors of Mamba2 or Hugging Face.


## Why?
- Don't have much time to properly test everything.
- Wanted a HF compatible version.
- Wanted to use any optimisation without the cuda wheels required by the original mamba repo.
- Less interested in hybrid attention variant --> needs [flash  attention](https://github.com/Dao-AILab/flash-attention) (due to RoPE embeds).


## Installation
I won't distribute a pypi package, but you can use it as package by cloning the repo and installing it at root:
```bash
git clone https://github.com/vasqu/mamba2-torch.git
cd mamba2-torch
pip install .
``` 
To use the "fastest" path, you need to install the [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) package separately.


## Usage
To use any `Mamba2` model you need to convert it first to a compatible format. For this purpose you can use this [script](./scripts/convert_mamba2.sh) after installing this package.
```bash
# example usage to download mamba2-130m
# 1st argument = parameter count, 2nd argument = directory to save the converted model to
./convert_mamba2.sh 130m ../models
```

Now you can use the converted model the following way.
```python
from transformers import AutoTokenizer
from mamba2_torch import Mamba2Model, Mamba2ForCausalLM, Mamba2Config

mamba2_hf_path = "<path-to-converted-model>"

# you can optionally disable the trition kernels if you want to
# this automatically happens when you use it on cpu
config = Mamba2Config.from_pretrained(mamba2_hf_path, local_files_only=True)
config.use_triton_kernels = False

# load model
model = Mamba2ForCausalLM.from_pretrained(mamba2_hf_path, config=config, local_files_only=True).to("cuda")

# the tokenizer is the same across any of the mamba models
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"].to("cuda")

# expected output: `["Hey how are you doing?\n\nI'm in the middle of a project"]`
out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```


## Some (maybe interesting) notes
- Most work goes to the original [mamba repo](https://github.com/state-spaces/mamba). They did the heavy work, give them your flowers.
- [ssd_minimal](./tests/ssd_minimal.py) is a small script based on the original script provided by Tri Dao and Albert Gu (see [here](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py)) and modified to work on any sequence length and with the `D` residual connection. A small test that checks roughly equal outputs is [over here](./tests/TestSSDMinimal.py).
- Possible [AMD support](https://github.com/state-spaces/mamba/pull/359) is in the works.
- You can get the last state of the SSD block via the `output_last_ssm_states` flag. This can be either passed down in the forward call itself or by modifying the config beforehand as done in the [example](#usage); just do `config.output_last_ssm_states = True` instead.
- Careful: With the following feature you won't be able to use the cache anymore; make sure to turn caching off!
    - You can pass initial states to the SSD blocks. This requires a list of the ssm state tensors of shape = `(batch_size, num_heads, head_dim, state_size)`. The list is as long as the number of hidden blocks. If you only want to pass initial states to a set of them, then fill those you don't want to pass it to to `None`.
- There are still some issues I'm not so sure of myself:
    - [Compiling](https://github.com/vasqu/mamba2-torch/issues/1#issue-2349175830) doesn't seem to work on my end which would boost the performance of triton kernels even more.
    - [NaN losses](https://github.com/vasqu/mamba2-torch/issues/2#issue-2349255152) seem to be fixed but you have to make sure that `( (d_model * expand) / headdim ) % 8 == 0`.
- To properly utilize caching, you will need (at least) the pinned version in the [requirements.txt](requirements.txt) of the transformers library.
- Some optional parallelization options introduced in the original mamba2 repo have been left out:
    - Groups in Multi-input SSM
    - Parallelized linear layers
    - Imo insignificant kernels (e.g. RMSNorm)
- `tie_embedding_weights` flag in the config is probably enforced in any case. Not too interested in digging into this but open for PRs.


## Work this is based on
 ```bibtex
[1] Mamba2
@inproceedings{mamba2,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}

[2] Hugging Face
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
 ```
