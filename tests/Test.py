"""import torch
from transformers import AutoTokenizer
from src.mamba2_torch.modeling.modeling_mamba2 import Mamba2ForCausalLM, Mamba2Model


mamba2_130m_hf_path = "/datadisk1/av11/downloads/huggingface/models--mamba2-130m-hf"
model = Mamba2ForCausalLM.from_pretrained(mamba2_130m_hf_path, local_files_only=True).to("cuda")
model_2 = Mamba2Model.from_pretrained(mamba2_130m_hf_path, local_files_only=True).to("cuda")

print(f'Embedding has been successfully loaded: {torch.allclose(model.backbone.embeddings.weight, model_2.embeddings.weight)}\n')

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"].to("cuda")

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
"""


"""
y = mamba_chunk_scan_combined(
                x=rearrange(x, pattern="b l (h p) -> b l h p", p=self.head_dim),
                dt=dt,
                A=A,
                B=rearrange(B, pattern="b l n -> b l 1 n"),
                C=rearrange(C, pattern="b l n -> b l 1 n"),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                initial_states=initial_state,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=None,
                # split this into non-tuple format
                dt_min=0.0,
                dt_max=float("inf"),
                return_final_states=cached_start or return_final_state
            )
"""


from src.mamba2_torch.ops.ssd_combined import mamba_chunk_scan_combined
from tests.ssd_minimal import ssd_minimal_discrete
import torch
from einops import rearrange
import torch.nn.functional as F


torch.manual_seed(42)

## Dimensions
# Denoted (B, T, Q, D, P) in the paper
batch, seqlen, chunk_size, dim, headdim = 1, 2048, 64, 2048, 64
nheads = dim // headdim  # (H) in the paper
ngroups = 1 # (G) in the paper
dstate = 64  # (N) in the paper
dtype = torch.float32
device = "cuda"

x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
D = torch.randn(nheads, dtype=dtype, device=device)
initial_state = torch.randn(batch, nheads, headdim, dstate, dtype=dtype, device=device)

# Comparing fused version and minimal version
y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, initial_states=initial_state)
y_2 = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, initial_states=initial_state)
y_min, _ = ssd_minimal_discrete(x, dt, A, B, C, chunk_size, D=D, initial_states=rearrange(initial_state, "b n h d -> b 1 n h d"))
y_min_2, _ = ssd_minimal_discrete(x, dt, A, B, C, chunk_size, D=None, initial_states=rearrange(initial_state, "b n h d -> b 1 n h d"))

print(torch.allclose(y_2, y_min_2, atol=0.1, rtol=0.1))
print(torch.allclose(y_2, y_min_2, atol=0.01, rtol=0.01))
print(torch.allclose(y, y_min, atol=0.01, rtol=0.01))
print(torch.allclose(y, y_min, atol=0.01, rtol=0.01))
