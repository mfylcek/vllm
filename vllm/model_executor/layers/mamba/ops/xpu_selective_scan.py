# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
XPU Triton implementation of selective_scan_fwd for Mamba SSM models.

This implements the same algorithm as the CUDA kernel in
csrc/mamba/mamba_ssm/selective_scan_fwd.cu but using Triton,
enabling it to run on Intel XPU via triton-xpu.

The selective scan is a parallel prefix scan implementing the SSM recurrence:
  state[t] = exp(A * delta[t]) * state[t-1] + B[t] * delta[t] * u[t]
  out[t]   = sum(state[t] * C[t]) + D * u[t]

Tensor layouts (varlen mode):
  u:     (dim, total_length)
  delta: (dim, total_length)
  A:     (dim, dstate)
  B:     (n_groups, dstate, total_length) - variable B
  C:     (n_groups, dstate, total_length) - variable C
  D:     (dim,)
  z:     (dim, total_length) - optional gating
  ssm_states: (num_cache_entries, dim, dstate)

Tensor layouts (batch mode):
  u:     (batch, dim, seqlen)
  delta: (batch, dim, seqlen)
  A:     (dim, dstate)
  B:     (batch, n_groups, dstate, seqlen) - variable B
  C:     (batch, n_groups, dstate, seqlen) - variable C
  D:     (dim,)
  z:     (batch, dim, seqlen) - optional gating
  ssm_states: (num_cache_entries, dim, dstate)
"""

import torch

from vllm.triton_utils import tl, triton

LOG2E = 1.4426950408889634


@triton.jit
def _selective_scan_fwd_kernel(
    # Pointers
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    delta_bias_ptr,
    out_ptr,
    out_z_ptr,
    ssm_states_ptr,
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    block_idx_first_scheduled_ptr,
    block_idx_last_scheduled_ptr,
    initial_state_idx_ptr,
    cu_chunk_seqlen_ptr,
    last_chunk_indices_ptr,
    # Dimensions
    batch: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    n_groups: tl.constexpr,
    seqlen,
    null_block_id,
    block_size,
    # Strides for u: (batch_or_dim, dim_or_seqlen, seqlen_or_1)
    stride_u_batch,
    stride_u_dim,
    # Strides for delta
    stride_delta_batch,
    stride_delta_dim,
    # Strides for A: (dim, dstate)
    stride_A_dim,
    stride_A_dstate,
    # Strides for B
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    # Strides for C
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    # Strides for out
    stride_out_batch,
    stride_out_dim,
    # Strides for z
    stride_z_batch,
    stride_z_dim,
    # Strides for out_z
    stride_out_z_batch,
    stride_out_z_dim,
    # Strides for ssm_states: (batch/cache, dim, dstate)
    stride_ssm_batch,
    stride_ssm_dim,
    stride_ssm_dstate,
    # Cache indices stride (for 2D cache_indices in APC mode)
    stride_cache_indices,
    # Flags
    delta_softplus: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    CACHE_ENABLED: tl.constexpr,
    HAS_DELTA_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_DSTATE: tl.constexpr,
):
    # Grid: (batch, dim)
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)

    # Determine sequence boundaries
    if IS_VARLEN:
        seq_start = tl.load(query_start_loc_ptr + pid_batch)
        seq_end = tl.load(query_start_loc_ptr + pid_batch + 1)
        cur_seqlen = seq_end - seq_start
    else:
        seq_start = pid_batch
        cur_seqlen = seqlen

    # Compute group index for B and C
    dim_ngroups_ratio = dim // n_groups
    group_id = pid_dim // dim_ngroups_ratio

    # Resolve cache index
    if HAS_CACHE_INDICES:
        if CACHE_ENABLED:
            init_state_idx = tl.load(initial_state_idx_ptr + pid_batch)
            cache_index = tl.load(
                cache_indices_ptr + pid_batch * stride_cache_indices
                + init_state_idx
            )
        else:
            cache_index = tl.load(cache_indices_ptr + pid_batch)
        # Skip null blocks (padding)
        if cache_index == null_block_id:
            return
    else:
        cache_index = pid_batch

    # Check if this batch entry has an initial state
    has_init_state = False
    if HAS_INITIAL_STATE:
        has_init_state = tl.load(has_initial_state_ptr + pid_batch)

    # Load D and delta_bias for this dim
    D_val = 0.0
    if HAS_D:
        D_val = tl.load(D_ptr + pid_dim).to(tl.float32)

    delta_bias_val = 0.0
    if HAS_DELTA_BIAS:
        delta_bias_val = tl.load(delta_bias_ptr + pid_dim).to(tl.float32)

    # Load A values for this dim: A[pid_dim, 0:dstate]
    offs_dstate = tl.arange(0, BLOCK_DSTATE)
    dstate_mask = offs_dstate < dstate
    A_vals = tl.load(
        A_ptr + pid_dim * stride_A_dim + offs_dstate * stride_A_dstate,
        mask=dstate_mask,
        other=0.0,
    ).to(tl.float32)
    # Pre-multiply A with LOG2E for exp2f
    A_log2e = A_vals * LOG2E

    # Determine initial load cache slot
    if CACHE_ENABLED and HAS_CACHE_INDICES:
        init_state_idx_val = tl.load(initial_state_idx_ptr + pid_batch)
        load_cache_slot = tl.load(
            cache_indices_ptr + pid_batch * stride_cache_indices
            + init_state_idx_val
        ).to(tl.int64)
    else:
        load_cache_slot = cache_index

    # Load initial SSM state: ssm_states[load_cache_slot, pid_dim, 0:dstate]
    state = tl.zeros([BLOCK_DSTATE], dtype=tl.float32)
    if has_init_state:
        if CACHE_ENABLED and HAS_CACHE_INDICES:
            state = tl.load(
                ssm_states_ptr
                + load_cache_slot * stride_ssm_batch
                + pid_dim * stride_ssm_dim
                + offs_dstate * stride_ssm_dstate,
                mask=dstate_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            state = tl.load(
                ssm_states_ptr
                + cache_index * stride_ssm_batch
                + pid_dim * stride_ssm_dim
                + offs_dstate * stride_ssm_dstate,
                mask=dstate_mask,
                other=0.0,
            ).to(tl.float32)

    # Determine chunk boundaries
    effective_block_size = block_size if CACHE_ENABLED else 2048

    # Get APC chunk metadata if available
    has_chunk_meta = CACHE_ENABLED and HAS_CACHE_INDICES
    first_chunk_idx = 0
    n_chunks = (cur_seqlen + effective_block_size - 1) // effective_block_size
    if n_chunks == 0:
        n_chunks = 1

    # Base pointers for u, delta, B, C, z, out
    if IS_VARLEN:
        u_base = u_ptr + pid_dim * stride_u_dim
        delta_base = delta_ptr + pid_dim * stride_delta_dim
        B_base = B_ptr + group_id * stride_B_group
        C_base = C_ptr + group_id * stride_C_group
        out_base = out_ptr + pid_dim * stride_out_dim
        if HAS_Z:
            z_base = z_ptr + pid_dim * stride_z_dim
            out_z_base = out_z_ptr + pid_dim * stride_out_z_dim
    else:
        u_base = (
            u_ptr + seq_start * stride_u_batch + pid_dim * stride_u_dim
        )
        delta_base = (
            delta_ptr + seq_start * stride_delta_batch
            + pid_dim * stride_delta_dim
        )
        B_base = (
            B_ptr + seq_start * stride_B_batch + group_id * stride_B_group
        )
        C_base = (
            C_ptr + seq_start * stride_C_batch + group_id * stride_C_group
        )
        out_base = (
            out_ptr + seq_start * stride_out_batch
            + pid_dim * stride_out_dim
        )
        if HAS_Z:
            z_base = (
                z_ptr + seq_start * stride_z_batch
                + pid_dim * stride_z_dim
            )
            out_z_base = (
                out_z_ptr + seq_start * stride_out_z_batch
                + pid_dim * stride_out_z_dim
            )

    # Process the sequence element by element.
    # This is the sequential scan - each element depends on the previous state.
    tokens_processed = 0
    for chunk in range(n_chunks):
        chunk_tokens = tl.minimum(
            effective_block_size, cur_seqlen - tokens_processed
        )
        if chunk_tokens <= 0:
            break

        for t in range(effective_block_size):
            if tokens_processed + t >= cur_seqlen:
                break

            seq_pos = seq_start + tokens_processed + t if IS_VARLEN else (
                tokens_processed + t
            )

            # Load u[t] and delta[t]
            if IS_VARLEN:
                u_val = tl.load(u_base + seq_pos).to(tl.float32)
                delta_val = tl.load(delta_base + seq_pos).to(tl.float32)
            else:
                u_val = tl.load(u_base + (tokens_processed + t)).to(
                    tl.float32
                )
                delta_val = tl.load(
                    delta_base + (tokens_processed + t)
                ).to(tl.float32)

            # Apply delta bias and softplus
            delta_val = delta_val + delta_bias_val
            if delta_softplus:
                delta_val = tl.where(
                    delta_val <= 20.0,
                    tl.math.log(tl.math.exp(delta_val) + 1.0),
                    delta_val,
                )

            delta_u = delta_val * u_val

            # Load B[t] and C[t] for all dstate positions
            if IS_VARLEN:
                B_t = tl.load(
                    B_base + offs_dstate * stride_B_dstate + seq_pos,
                    mask=dstate_mask,
                    other=0.0,
                ).to(tl.float32)
                C_t = tl.load(
                    C_base + offs_dstate * stride_C_dstate + seq_pos,
                    mask=dstate_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                B_t = tl.load(
                    B_base
                    + offs_dstate * stride_B_dstate
                    + (tokens_processed + t),
                    mask=dstate_mask,
                    other=0.0,
                ).to(tl.float32)
                C_t = tl.load(
                    C_base
                    + offs_dstate * stride_C_dstate
                    + (tokens_processed + t),
                    mask=dstate_mask,
                    other=0.0,
                ).to(tl.float32)

            # SSM recurrence:
            #   state = exp(A * delta) * state + B * delta * u
            dA = tl.math.exp2(A_log2e * delta_val)
            state = state * dA + B_t * delta_u

            # Output: y = sum(state * C) + D * u
            out_val = tl.sum(state * C_t, axis=0) + D_val * u_val

            # Store output
            if IS_VARLEN:
                tl.store(out_base + seq_pos, out_val.to(out_base.dtype.element_ty))
            else:
                tl.store(
                    out_base + (tokens_processed + t),
                    out_val.to(out_base.dtype.element_ty),
                )

            # If z (gating) is present: out_z = out * z * sigmoid(z)
            if HAS_Z:
                if IS_VARLEN:
                    z_val = tl.load(z_base + seq_pos).to(tl.float32)
                    gated_out = out_val * z_val * tl.sigmoid(z_val)
                    tl.store(
                        out_z_base + seq_pos,
                        gated_out.to(out_z_base.dtype.element_ty),
                    )
                else:
                    z_val = tl.load(
                        z_base + (tokens_processed + t)
                    ).to(tl.float32)
                    gated_out = out_val * z_val * tl.sigmoid(z_val)
                    tl.store(
                        out_z_base + (tokens_processed + t),
                        gated_out.to(out_z_base.dtype.element_ty),
                    )

        tokens_processed += chunk_tokens

        # Store SSM state at end of chunk
        if CACHE_ENABLED and HAS_CACHE_INDICES:
            block_idx_last_val = tl.load(
                block_idx_last_scheduled_ptr + pid_batch
            )
            store_slot = tl.load(
                cache_indices_ptr + pid_batch * stride_cache_indices
                + block_idx_last_val
            ).to(tl.int64)
            tl.store(
                ssm_states_ptr
                + store_slot * stride_ssm_batch
                + pid_dim * stride_ssm_dim
                + offs_dstate * stride_ssm_dstate,
                state.to(ssm_states_ptr.dtype.element_ty),
                mask=dstate_mask,
            )

    # Store final SSM state (non-APC mode)
    if not CACHE_ENABLED or not HAS_CACHE_INDICES:
        tl.store(
            ssm_states_ptr
            + cache_index * stride_ssm_batch
            + pid_dim * stride_ssm_dim
            + offs_dstate * stride_ssm_dstate,
            state.to(ssm_states_ptr.dtype.element_ty),
            mask=dstate_mask,
        )


def xpu_selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    delta_bias: torch.Tensor | None,
    delta_softplus: bool,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    ssm_states: torch.Tensor,
    null_block_id: int,
    block_size: int = 1024,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    cu_chunk_seqlen: torch.Tensor | None = None,
    last_chunk_indices: torch.Tensor | None = None,
):
    """
    XPU Triton implementation of selective_scan_fwd.
    Writes output in-place to delta (or z if z is provided).
    """
    varlen = query_start_loc is not None
    sizes = u.sizes()

    if varlen:
        batch_size = query_start_loc.shape[0] - 1
        dim = sizes[0]
        seqlen = sizes[1]
    else:
        batch_size = sizes[0]
        dim = sizes[1]
        seqlen = sizes[2]

    dstate = A.size(1)
    n_groups = B.size(0) if varlen else B.size(1)

    # Output is written in-place to delta; if z is present, gated output
    # is written to z.
    out = delta

    has_z = z is not None
    cache_enabled = (
        cache_indices is not None
        and block_idx_first_scheduled_token is not None
    )

    # Cache indices stride for 2D case
    cache_indices_stride = (
        cache_indices.stride(0)
        if cache_indices is not None and cache_indices.dim() == 2
        else 0
    )

    BLOCK_DSTATE = triton.next_power_of_2(dstate)

    grid = (batch_size, dim)

    _selective_scan_fwd_kernel[grid](
        # Pointers
        u,
        delta,
        A,
        B,
        C,
        D if D is not None else u,  # dummy, won't be read
        z if z is not None else u,  # dummy, won't be read
        delta_bias if delta_bias is not None else u,  # dummy
        out,
        z if has_z else u,  # out_z = z (in-place) or dummy
        ssm_states,
        query_start_loc if query_start_loc is not None else u,  # dummy
        (
            cache_indices if cache_indices is not None else u
        ),  # dummy
        (
            has_initial_state if has_initial_state is not None else u
        ),  # dummy
        (
            block_idx_first_scheduled_token
            if block_idx_first_scheduled_token is not None
            else u
        ),
        (
            block_idx_last_scheduled_token
            if block_idx_last_scheduled_token is not None
            else u
        ),
        (
            initial_state_idx if initial_state_idx is not None else u
        ),
        (
            cu_chunk_seqlen if cu_chunk_seqlen is not None else u
        ),
        (
            last_chunk_indices if last_chunk_indices is not None else u
        ),
        # Dimensions
        batch_size,
        dim,
        dstate,
        n_groups,
        seqlen,
        null_block_id,
        block_size,
        # Strides for u
        u.stride(0) if not varlen else 0,
        u.stride(0) if varlen else u.stride(1),
        # Strides for delta
        delta.stride(0) if not varlen else 0,
        delta.stride(0) if varlen else delta.stride(1),
        # Strides for A
        A.stride(0),
        A.stride(1),
        # Strides for B
        B.stride(0) if not varlen else 0,
        B.stride(0) if varlen else B.stride(1),
        B.stride(-2),
        # Strides for C
        C.stride(0) if not varlen else 0,
        C.stride(0) if varlen else C.stride(1),
        C.stride(-2),
        # Strides for out
        out.stride(0) if not varlen else 0,
        out.stride(0) if varlen else out.stride(1),
        # Strides for z
        (z.stride(0) if not varlen else 0) if has_z else 0,
        (z.stride(0) if varlen else z.stride(1)) if has_z else 0,
        # Strides for out_z (same as z since in-place)
        (z.stride(0) if not varlen else 0) if has_z else 0,
        (z.stride(0) if varlen else z.stride(1)) if has_z else 0,
        # Strides for ssm_states
        ssm_states.stride(0),
        ssm_states.stride(1),
        ssm_states.stride(2),
        # Cache indices stride
        cache_indices_stride,
        # Flags
        delta_softplus,
        D is not None,
        has_z,
        varlen,
        cache_indices is not None,
        has_initial_state is not None,
        cache_enabled,
        delta_bias is not None,
        # Block sizes
        BLOCK_DSTATE,
    )
