import torch
import torch.nn.functional as F


def pad_by_size(x, pad_size):
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(x.shape) == 4 \
        else (0, 0, 0, pad_size, 0, 0)

    return F.pad(x, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = x[..., None].expand(*x.size(), T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(X, dt, A, B, C, block_len, D=None, initial_states=None):
    """
    Arguments:
        X:  (batch, length, n_heads, d_head)
        dt: (batch, length, n_heads)
        A:  (n_heads)
        B:  (batch, length, n_heads, d_state)
        C:  (batch, length, n_heads, d_state)
    Return:
        Y:  (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype

    seq_len = X.shape[1]
    pad_size = (block_len - seq_len % block_len) % block_len
    #assert X.shape[1] % block_len == 0

    # (Optional) D skip connection preparing
    if D is not None:
        skip = D.unsqueeze(-1) * pad_by_size(X, pad_size)

    # Discretize X and A
    X = X * dt.unsqueeze(-1)
    A = A * dt

    # Rearrange into blocks/chunks
    X, A, B, C = [reshape_into_chunks(x, pad_size, block_len) for x in (X, A, B, C)]

    A = A.permute(0, 3, 1, 2)  # "b c l h -> b h c l"
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    C_head_reshape = C.permute(0, -2, 1, 2, -1)  # bclhn -> bhcln
    B_head_reshape = B.permute(0, -2, 1, 2, -1)  # bcshn -> bhcsn
    G = (C_head_reshape[..., None, :] * B_head_reshape[..., None, :, :]).sum(dim=-1)  # "bhcl1n,bhc1sn -> bhcls"

    L = torch.exp(segsum(A))
    M = G * L  # "bhcls,bhcls -> bhcls"

    Y_diag = (M.permute(0, 2, -1, -2, 1)[..., None] * X[..., None, :, :]).sum(dim=2)  # "bcslh1,bcs1hp -> bclhp"

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]  # "bclhn,bclh1 -> bclhn
    states = (B_decay[..., None, :] * X[..., None]).sum(dim=2)   # "bclh1n,bclhp1 -> bchpn"

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    decay_chunk = decay_chunk.transpose(1, 3)                              # "bhzc -> bczh"
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)  # "bczh11,bc1hpn -> bzhpn"
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    state_decay_out = state_decay_out.permute(0, -2, -1, 1)                 # "bhcl -> bclh"
    Y_off_states = (C[..., None, :] * states[:, :, None, ...]).sum(dim=-1)  # "bclh1n,bc1hpn -> bclhp"
    Y_off = Y_off_states * state_decay_out[..., None]                       # "bclhp,bclh1 -> bclhp"

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    bsz, _, _, num_heads, head_dim = Y_diag.shape
    Y = (Y_diag + Y_off).reshape(bsz, -1, num_heads, head_dim)  # "bclhp -> b(c l)hp"

    # Add optional D residual
    if D is not None:
        Y = Y + skip

    if pad_size > 0:
        Y = Y[:, :seq_len, :, :]

    return Y, final_state
