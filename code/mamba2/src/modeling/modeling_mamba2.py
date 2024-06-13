"""PyTorch MAMBA2 model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from einops import rearrange, repeat
from src.ops.ssd_combined import mamba_chunk_scan_combined
from src.ops.selective_state_update import selective_state_update

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.utils.import_utils import is_causal_conv1d_available
from src.modeling.configuration_mamba2 import Mamba2Config


logger = logging.get_logger(__name__)

is_fast_path_available = False
if is_causal_conv1d_available():
    from src.ops.ssd_combined import mamba_split_conv1d_scan_combined
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    is_fast_path_available = True
else:
    mamba_split_conv1d_scan_combined = None
    causal_conv1d_fn, causal_conv1d_update = None, None


class Mamba2Cache:
    def __init__(self, config: Mamba2Config, batch_size: int, device=None, dtype=torch.float16):
        self.seq_offset = 0

        in_channels = config.intermediate_size + 2 * config.state_size
        self.conv_states = {
            i: torch.zeros(batch_size, in_channels, config.conv_kernel, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }

        self.ssm_states = {
            i: torch.zeros(batch_size, config.num_heads, config.head_dim, config.state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


class Mamba2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Mamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["Mamba2Block"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, Mamba2Mixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.config.num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True
            module.dt_bias._no_weight_decay = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.emb_initializer_range)
        elif isinstance(module, nn.Conv1d):
            if self.config.conv_initializer_range is not None:
                nn.init.uniform_(module.weight, -self.config.conv_initializer_range, self.config.conv_initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


class Mamba2Mixer(nn.Module):
    """
    Using the found relation to the attention mechanism under certain conditions (State-Space-Duality SSD),
    we use the Multi-input SSM which can be seen as a counterpart to the Multi-value Attention with analogues:
    - X ~= V
    - B ~= Q
    - C ~= K
    - A (1-SS(a)) ~= Attention Mask

    For an overview, see the mamba2 paper, section 6, figure 4.
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.chunk_size = config.chunk_size
        self.dt_min = config.time_step_limit[0]
        self.dt_max = config.time_step_limit[1]
        self.layer_idx = layer_idx
        self.use_bias = config.use_bias
        self.use_conv_bias = config.use_conv_bias

        # parallel projection of the input hidden states
        self.in_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=2 * (self.intermediate_size + self.ssm_state_size) + config.num_heads,
            bias=config.use_bias
        )

        conv1d_dim = self.intermediate_size + 2 * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=conv1d_dim,
            out_channels=conv1d_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=conv1d_dim,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # we only use a bias as parameter
        self.dt_bias = nn.Parameter(torch.rand(size=(config.num_heads,)))

        # scalar initialization of A, i.e. 1-Semi-Separable Matrix of A (== 1-SS(a))
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(*config.A_initializer_range)
        self.A_log = nn.Parameter(torch.log(A))

        # as D is a skip connection with A, it is also a scalar of the same shape as A
        self.D = nn.Parameter(torch.ones(self.num_heads))

        # residual normalization introduced for instability, see section 7 of the paper
        self.norm = Mamba2RMSNorm(
            self.intermediate_size, eps=1e-5, normalize=True
        )

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because `(causal_conv1d_fn, causal_conv1d_update)` is None. "
                "Falling back to the naive implementation. To install follow https://github.com/Dao-AILab/causal-conv1d"
            )

    def forward(self, hidden_states, initial_state=None, return_final_state=False, cache: Optional[Mamba2Cache] = None):
        # managing cache state
        if cache is not None:
            cached_start = cache.seq_offset == 0
            cached_forward = not cached_start
        else:
            cached_start = False
            cached_forward = False

        # supporting cached values as well as passing initial states but not both at the same time
        if initial_state is not None and cached_forward:
            raise ValueError("Subsequent caching and passing initial states is not possible at the same time!")

        # 1. Parallel projection for the input
        zxbcdt = self.in_proj(hidden_states)

        # 2-5. Combined into one triton kernel
        if self.training and cache is None and is_fast_path_available:
            y = mamba_split_conv1d_scan_combined(
                zxbcdt=zxbcdt,
                conv1d_weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                conv1d_bias=self.conv1d.bias,
                dt_bias=self.dt_bias,
                A=-torch.exp(self.A_log),
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=None,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.head_dim,
                ngroups=1,
                norm_before_gate=False,  # not the same as our variant's normalization var
                # split this into non-tuple format
                dt_min=0.0,
                dt_max=float("inf"),
                initial_states=initial_state,
                return_final_states=return_final_state
            )
            last_state = None
            if return_final_state:
                y, last_state = y
            return y, last_state

        # reconstructing the necessary vars
        d_mlp = (zxbcdt.shape[-1] - 2 * self.intermediate_size - 2 * self.ssm_state_size - self.num_heads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.intermediate_size, self.intermediate_size + 2 * self.ssm_state_size, self.num_heads],
            dim=-1
        )

        # 2. Causal convolution for partial set of variables ("input", B, C)
        xBC = self._conv1d(
            xBC=xBC, seq_len=hidden_states.shape[1],
            cache=cache, cached_start=cached_start, cached_forward=cached_forward
        )

        # reconstruct causal convolution vars
        x, B, C = torch.split(
            xBC, [self.intermediate_size, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        # 3. State Space Duality (SSD) with triton kernel(s)
        y, last_state = self._ssd(
            x=x, B=B, C=C, dt=dt,
            initial_state=initial_state, return_final_state=return_final_state,
            cache=cache, cached_start=cached_start, cached_forward=cached_forward
        )

        # 4. Gate normalization introduced for instability, see section 7 of the paper
        y = self.norm(y, residual=z)
        if d_mlp > 0:
            y = torch.cat([self.act(z0) * x0, y], dim=-1)

        # 5. Out projecting
        y = self.out_proj(y)

        return y, last_state

    def _conv1d(self, xBC, seq_len, cache, cached_start, cached_forward):
        # init cache with first "real" values
        if cached_start:
            xBC_t = rearrange(xBC, "b l d -> b d l")
            cache.conv_states[self.layer_idx].copy_(nn.functional.pad(xBC_t, (self.conv_kernel_size - xBC_t.shape[-1], 0)))

        if is_fast_path_available:
            if cached_forward:
                xBC = causal_conv1d_update(
                    xBC,
                    cache.conv_states[self.layer_idx],
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
        else:
            if cached_forward:
                cache.conv_states[self.layer_idx].copy_(torch.roll(cache.conv_states[self.layer_idx], shifts=-1, dims=-1))
                cache.conv_states[self.layer_idx][:, :, -1] = xBC
                xBC = torch.sum(cache.conv_states[self.layer_idx] * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
                if self.conv1d.bias is not None:
                    xBC = xBC + self.conv1d.bias
                xBC = self.act(xBC)
            else:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2))[..., :seq_len].transpose(1, 2)
                )

        return xBC

    def _ssd(self, x, B, C, dt, initial_state, return_final_state, cache, cached_start, cached_forward):
        # discretize 1-SS(a)
        A = -torch.exp(self.A_log) if not cached_forward else -torch.exp(self.A_log.float())

        last_state = None
        if not cached_forward:
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
            if cached_start or return_final_state:
                y, last_state = y
                if cached_start:
                    cache.ssm_states[self.layer_idx].copy_(last_state)

            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            A = repeat(A, "h -> h p n", p=self.head_dim, n=self.ssm_state_size).to(dtype=torch.float32)
            dt = repeat(dt, "b 1 h -> b h p", p=self.head_dim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.head_dim)
            D = repeat(self.D, "h -> h p", p=self.head_dim)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.head_dim)
            y = selective_state_update(
                cache.ssm_states[self.layer_idx], x_reshaped, dt, A, B, C, D, z=None,
                dt_bias=dt_bias, dt_softplus=True
            )
            if return_final_state:
                last_state = cache.ssm_states[self.layer_idx].detach().clone()

            y = rearrange(y, "b h p -> b 1 (h p)")

        return y, last_state


class Mamba2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, normalize=False):
        """
        Mamba2RMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm but with optional residual normalizing
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.normalize = normalize

    def forward(self, hidden_states, residual=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # residual normalization introduced for instability, see section 7 of the paper
        if residual is not None and self.normalize:
            hidden_states = hidden_states * nn.functional.silu(residual.to(torch.float32))

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states * self.weight

        return hidden_states.to(input_dtype)


class Mamba2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(config, layer_idx=layer_idx)

    def forward(self, hidden_states, initial_state=None, return_final_state=False, cache: Optional[Mamba2Cache] = None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states, last_state = self.mixer(hidden_states, initial_state=initial_state, return_final_state=return_final_state, cache=cache)
        hidden_states = residual + hidden_states
        return hidden_states, last_state


@dataclass
class Mamba2Output(ModelOutput):
    """
    Class for the MAMBA2 model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`Mamba2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the last state returned by the SSD call of the State Space Machine, and the Causal Convolutional states.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        last_ssm_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_last_ssm_states=True` is passed or when `config.output_last_ssm_states=True`):
            Tuple of `torch.FloatTensor` (one for the last state of the ssd block each) of shape `(batch_size, num_heads, head_dim, ssm_state_size)`.

            Last SSM-states of the model at the final state of an SSD block.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    last_ssm_states: Optional[Tuple[torch.FloatTensor]] = None


class Mamba2Model(Mamba2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            cache_params: Optional[Mamba2Cache] = None,
            use_cache: Optional[bool] = None,
            initial_states: Optional[List[torch.FloatTensor]] = None,
            output_hidden_states: Optional[bool] = None,
            output_last_ssm_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, Mamba2Output]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_last_ssm_states = (
            output_last_ssm_states if output_last_ssm_states is not None else self.config.output_last_ssm_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = Mamba2Cache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        initial_states = [None] * self.config.num_hidden_layers if initial_states is None else initial_states
        if len(initial_states) != self.config.num_hidden_layers:
            raise ValueError(
                "Initial states have been passed but not for all layers making it ambiguous. "
                "To ensure correctness, fill layers without an initial state with None."
            )
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_last_ssm_states = () if output_last_ssm_states else None
        for mixer_block, initial_state in zip(self.layers, initial_states):
            if self.gradient_checkpointing and self.training:
                out = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, initial_state, output_last_ssm_states, cache_params)
            else:
                out = mixer_block(hidden_states, initial_state=initial_state, return_final_state=output_last_ssm_states, cache=cache_params)

            hidden_states = out[0]
            last_state = out[1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_last_ssm_states:
                all_last_ssm_states = all_last_ssm_states + (last_state,)

        if use_cache:
            cache_params.seq_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
            last_ssm_states=all_last_ssm_states
        )


@dataclass
class Mamba2CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`Mamba2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the last state returned by the SSD call of the State Space Machine, and the Causal Convolutional states.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        last_ssm_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_last_ssm_states=True` is passed or when `config.output_last_ssm_states=True`):
            Tuple of `torch.FloatTensor` (one for the last state of the ssd block each) of shape `(batch_size, num_heads, head_dim, ssm_state_size)`.

            Last SSM-states of the model at the final state of an SSD block.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    last_ssm_states: Optional[Tuple[torch.FloatTensor]] = None


class Mamba2ForCausalLM(Mamba2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight", "backbone.embeddings.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = Mamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._tie_weights()

        # Initialize weights and apply final processing
        self.post_init()

    def _tie_weights(self):
        # probably overwritten by `_tied_weights_keys` but just to be sure
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
            self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            inputs_embeds=None,
            use_cache=None,
            cache_params: Optional[Mamba2Cache] = None,
            **kwargs,
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
            }
        )
        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            cache_params: Optional[Mamba2Cache] = None,
            initial_states: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_last_ssm_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, Mamba2CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            initial_states=initial_states,
            output_hidden_states=output_hidden_states,
            output_last_ssm_states=output_last_ssm_states,
            return_dict=return_dict,
        )
        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Mamba2CausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
            last_ssm_states=mamba_outputs.last_ssm_states
        )
