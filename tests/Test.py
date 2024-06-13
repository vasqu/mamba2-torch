import torch
from transformers import AutoTokenizer
from mamba2_torch.modeling.modeling_mamba2 import Mamba2ForCausalLM, Mamba2Model


mamba2_130m_hf_path = "/datadisk1/av11/downloads/huggingface/models--mamba2-130m-hf"
model = Mamba2ForCausalLM.from_pretrained(mamba2_130m_hf_path, local_files_only=True).to("cuda")
model_2 = Mamba2Model.from_pretrained(mamba2_130m_hf_path, local_files_only=True).to("cuda")

print(f'Embedding has been successfully loaded: {torch.allclose(model.backbone.embeddings.weight, model_2.embeddings.weight)}\n')

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"].to("cuda")

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
