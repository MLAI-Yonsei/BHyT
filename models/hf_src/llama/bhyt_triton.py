"""Triton kernels for BHyT decode-time pre/post normalization.

These are opt-in fused kernels meant to replace the per-norm Python op chain
(variance → rsqrt → tanh(λ·rstd·x) → ·w) with a single Triton kernel per norm
call. Designed for the tiny decode shape (B, 1, H) where launch overhead
dominates eager throughput.

Activation: set env var `BHYT_TRITON_KERNEL=1`. If unset (default) the original
eager Python helpers in `modeling_llama.py` are used unchanged.

Validity: covers only the `var_approx_method in (None, "baseline")` path with
`use_gamma_var=False` and `_use_exact_var=False`. For any other config the
caller must fall back to eager.

Author: 2026-05-18 (BHyT throughput optimization)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh as _triton_tanh


# -------------- Kernel: pre-norm (computes rstd1 + writes var_x) -------------

@triton.jit
def _bhyt_pre_norm_kernel(
    h_ptr, w_ptr, lam_ptr,
    out_ptr, var_x_out_ptr,
    INV_KAPPA: tl.constexpr,
    EPS: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    h_off = row * n_cols + cols
    x = tl.load(h_ptr + h_off, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / n_cols
    rstd = INV_KAPPA / tl.sqrt(var + EPS)
    lam = tl.load(lam_ptr).to(tl.float32)
    alpha = lam * rstd

    z = alpha * x
    t = _triton_tanh(z)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = w * t

    tl.store(out_ptr + h_off, out, mask=mask)
    tl.store(var_x_out_ptr + row, var)


# -------------- Kernel: post-norm (reads var_x, adds c_correction) -----------

@triton.jit
def _bhyt_post_norm_kernel(
    h_ptr, w_ptr, lam_ptr, var_x_in_ptr, c_corr_ptr,
    out_ptr,
    INV_KAPPA: tl.constexpr,
    EPS: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    h_off = row * n_cols + cols
    x = tl.load(h_ptr + h_off, mask=mask, other=0.0).to(tl.float32)

    var_x_prev = tl.load(var_x_in_ptr + row).to(tl.float32)
    c_corr = tl.load(c_corr_ptr).to(tl.float32)
    var = var_x_prev + c_corr
    rstd = INV_KAPPA / tl.sqrt(var + EPS)
    lam = tl.load(lam_ptr).to(tl.float32)
    alpha = lam * rstd

    z = alpha * x
    t = _triton_tanh(z)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = w * t

    tl.store(out_ptr + h_off, out, mask=mask)


# -------------- Python wrappers ----------------------------------------------

_EPS = 1e-6


def bhyt_pre_norm_triton(hidden_states: torch.Tensor,
                         lam: torch.Tensor,
                         weight: torch.Tensor,
                         inv_kappa: float):
    """Fused rstd1 + input BHyT norm.

    Args:
        hidden_states: (B, S, H), fp16/bf16, contiguous.
        lam: scalar tensor (any dtype, cast to fp32 inside).
        weight: (H,) fp16/bf16.
        inv_kappa: python float.

    Returns:
        out: (B, S, H), same dtype as hidden_states.
        var_x: (B, S, 1) fp32.
    """
    assert hidden_states.is_cuda and weight.is_cuda
    h = hidden_states.contiguous()
    B, S, H = h.shape
    n_rows = B * S
    out = torch.empty_like(h)
    var_x = torch.empty((B, S, 1), dtype=torch.float32, device=h.device)

    BLOCK = triton.next_power_of_2(H)
    _bhyt_pre_norm_kernel[(n_rows,)](
        h.view(n_rows, H), weight, lam.view(()).contiguous() if lam.ndim else lam,
        out.view(n_rows, H), var_x.view(n_rows),
        INV_KAPPA=float(inv_kappa),
        EPS=_EPS,
        n_cols=H,
        BLOCK=BLOCK,
        num_warps=8 if H >= 4096 else 4,
    )
    return out, var_x


def bhyt_post_norm_triton(hidden_states: torch.Tensor,
                          var_x: torch.Tensor,
                          c_correction: torch.Tensor,
                          lam: torch.Tensor,
                          weight: torch.Tensor,
                          inv_kappa: float):
    """Fused rstd2 + post BHyT norm. baseline var-approx only.

    Args:
        hidden_states: (B, S, H), fp16/bf16.
        var_x: (B, S, 1) fp32 from pre-norm.
        c_correction: scalar tensor (fp32) = c_attn * (lam²/κ²).
        lam: scalar tensor.
        weight: (H,).
        inv_kappa: python float.

    Returns:
        out: (B, S, H), same dtype as hidden_states.
    """
    h = hidden_states.contiguous()
    B, S, H = h.shape
    n_rows = B * S
    out = torch.empty_like(h)
    BLOCK = triton.next_power_of_2(H)

    _bhyt_post_norm_kernel[(n_rows,)](
        h.view(n_rows, H), weight,
        lam.view(()).contiguous() if lam.ndim else lam,
        var_x.view(n_rows),
        c_correction.view(()).contiguous() if c_correction.ndim else c_correction,
        out.view(n_rows, H),
        INV_KAPPA=float(inv_kappa),
        EPS=_EPS,
        n_cols=H,
        BLOCK=BLOCK,
        num_warps=8 if H >= 4096 else 4,
    )
    return out
