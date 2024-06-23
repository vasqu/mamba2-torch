from transformers import AutoTokenizer
from mamba2_torch import Mamba2Model, Mamba2Config, Mamba2ForCausalLM


device = "cuda"
mamba2_hf_path = "/datadisk1/av11/coding/github/mamba2-torch/test/mamba2-130m"

config = Mamba2Config.from_pretrained(mamba2_hf_path, local_files_only=True)
config.use_triton_kernels = False
config.max_sequence_chunk = 2

model = Mamba2ForCausalLM.from_pretrained(mamba2_hf_path, config=config, local_files_only=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(mamba2_hf_path, local_files_only=True)

input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"].to(device)

out = model.generate(input_ids, max_new_tokens=10, use_cache=False)
print(tokenizer.batch_decode(out))
