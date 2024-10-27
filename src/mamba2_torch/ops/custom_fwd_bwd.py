import torch


def custom_amp_decorator(dec, cuda_amp_deprecated):
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = "cuda"
        return dec(*args, **kwargs)
    return decorator


if hasattr(torch.amp, "custom_fwd"):
    deprecated = True
    from torch.amp import custom_fwd, custom_bwd
else:
    deprecated = False
    from torch.cuda.amp import custom_fwd, custom_bwd

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)
