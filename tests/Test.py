from transformers import AutoTokenizer
from mamba2_torch import Mamba2Model, Mamba2Config, Mamba2ForCausalLM


mamba2_130m_hf_path = "/datadisk1/av11/downloads/huggingface/models--mamba2-130m-hf"
config = Mamba2Config.from_pretrained(mamba2_130m_hf_path, local_files_only=True)
config.use_triton_kernels = False
model = Mamba2ForCausalLM.from_pretrained(mamba2_130m_hf_path, config=config, local_files_only=True).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"].to("cuda")

out = model.generate(input_ids, max_new_tokens=10, use_cache=False)
print(tokenizer.batch_decode(out))
