"""MAMBA2 configuration"""

from transformers.configuration_utils import PretrainedConfig


class Mamba2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Mamba2Model`]. It is used to instantiate a MAMBA2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MAMBA2
    [state-spaces/mamba2-130m](https://huggingface.co/state-spaces/mamba2-130m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50280):
            Vocabulary size of the MAMBA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MambaModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        state_size (`int`, *optional*, defaults to 16): shape of the state space latents.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end of sentence token in the vocabulary.
        expand (`int`, *optional*, defaults to 2): Expanding factor used to determine the intermediate size.
        conv_kernel (`int`, *optional*, defaults to 4): Size of the convolution kernel.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the mixer block.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.1):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
        time_step_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        time_step_scale (`float`, *optional*, defaults to 1.0):
            Scale used used to scale `dt_proj.bias`.
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj.bias`.
        time_step_init_scheme (`float`, *optional*, defaults to `"random"`):
            Init scheme used for `dt_proj.weight`. Should be one of `["random","uniform"]`
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether or not to rescale `out_proj` weights when initializing.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the cache should be used.
    """

    model_type = "mamba2"

    def __init__(
        self,
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
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.intermediate_size = int(expand * self.hidden_size)
        self.num_heads = self.intermediate_size // self.head_dim
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.emb_initializer_range = emb_initializer_range
        self.conv_initializer_range = conv_initializer_range
        self.A_initializer_range = A_initializer_range
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.time_step_limit = time_step_limit
        self.residual_in_fp32 = residual_in_fp32
        self.rescale_prenorm_residual = rescale_prenorm_residual

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
