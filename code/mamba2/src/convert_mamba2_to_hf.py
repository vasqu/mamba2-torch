vocab_size=50280,
pad_token_id=0,
bos_token_id=0,
eos_token_id=0,
hidden_size=768,
state_size=128,
head_dim=64,
chunk_size=256,
expand=2,
conv_kernel=4,
num_hidden_layers=24,
layer_norm_epsilon=1e-5,
use_bias=False,
use_conv_bias=True,
hidden_act="silu",
emb_initializer_range=0.02,
conv_initializer_range=None,
A_initializer_range=(1, 16),
time_step_min=0.001,
time_step_max=0.1,
time_step_floor=1e-4,
time_step_limit=(0.0, float("inf")),
residual_in_fp32=True,
rescale_prenorm_residual=True,
tie_embedding_weights=True,
output_last_ssm_states=False,
use_cache=True,