import torch
from transformers import AutoTokenizer
from src.mamba2_torch.modeling.modeling_mamba2 import Mamba2Model, Mamba2Config, Mamba2ForCausalLM


device = "cuda"
dtype = torch.bfloat16
mamba2_hf_path = "/datadisk1/av11/downloads/huggingface/mamba2-130m-av"

config = Mamba2Config.from_pretrained(mamba2_hf_path, local_files_only=True)
config.use_triton_kernels = False

model = Mamba2ForCausalLM.from_pretrained(mamba2_hf_path, config=config, local_files_only=True).to(device, dtype=dtype)
tokenizer = AutoTokenizer.from_pretrained(mamba2_hf_path, local_files_only=True)

input_ids = tokenizer(["Hey how are you doing?", "What is life?"], padding=True, return_tensors="pt")["input_ids"].to(device)

out = model.generate(input_ids, max_new_tokens=10, use_cache=True)
print(tokenizer.batch_decode(out))
