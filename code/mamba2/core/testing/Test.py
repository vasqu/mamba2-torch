from core.modeling.modeling_mamba2 import Mamba2ForCausalLM
from transformers import AutoTokenizer


mamba2_130m_hf_path = "/datadisk1/av11/downloads/huggingface/models--mamba2-130m-hf"
model = Mamba2ForCausalLM.from_pretrained(mamba2_130m_hf_path, local_files_only=True).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"].to("cuda")

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
