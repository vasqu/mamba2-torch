import torch
from einops import rearrange
import torch.nn.functional as F

from src.mamba2_torch.ops.ssd_combined import mamba_chunk_scan_combined
from tests.ssd_minimal import ssd_minimal_discrete as ssd_minimal_einops
from tests.ssd_minimal_no_einops import ssd_minimal_discrete as ssd_minimal_torch


torch.manual_seed(42)

## Dimensions
# Denoted (B, T, Q, D, P) in the paper
batch, seqlen, chunk_size, dim, headdim = 1, 100, 64, 2048, 64
nheads = dim // headdim  # (H) in the paper
ngroups = 8 # (G) in the paper
dstate = 64  # (N) in the paper
dtype = torch.float32
device = "cuda"

x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).repeat(1, 1, nheads // ngroups, 1)
C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).repeat(1, 1, nheads // ngroups, 1)
D = torch.randn(nheads, dtype=dtype, device=device)
initial_state = torch.randn(batch, nheads, headdim, dstate, dtype=dtype, device=device)

# Comparing fused version and minimal versions
y, l = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, initial_states=initial_state, return_final_states=True)
y_2 = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, initial_states=initial_state)

y_min_einops, l_min_einops = ssd_minimal_einops(x, dt, A, B, C, chunk_size, D=D, initial_states=rearrange(initial_state, "b n h d -> b 1 n h d"))
y_min_einops_2, _ = ssd_minimal_einops(x, dt, A, B, C, chunk_size, D=None, initial_states=rearrange(initial_state, "b n h d -> b 1 n h d"))

y_min_torch, l_min_torch = ssd_minimal_torch(x, dt, A, B, C, chunk_size, D=D, initial_states=rearrange(initial_state, "b n h d -> b 1 n h d"))
y_min_torch_2, _ = ssd_minimal_torch(x, dt, A, B, C, chunk_size, D=None, initial_states=rearrange(initial_state, "b n h d -> b 1 n h d"))


print(torch.allclose(l, l_min_einops, atol=0.01, rtol=0.01))
print(torch.allclose(l, l_min_torch, atol=0.01, rtol=0.01))
print(torch.allclose(y, y_min_einops, atol=0.01, rtol=0.01))
print(torch.allclose(y, y_min_torch, atol=0.01, rtol=0.01))
print(torch.allclose(y_2, y_min_einops_2, atol=0.01, rtol=0.01))
print(torch.allclose(y_2, y_min_torch_2, atol=0.01, rtol=0.01))
