# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Callable, Optional, Tuple, Union

import os

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from .configuration_llama import LlamaConfig

import math

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"

class BHyT_Star(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, lam_init: float = 2.0, learnable_lam: bool = False, off_gain: bool = False, inv_kappa: float = 0.1):
        super().__init__()
        self.eps = eps
        self.inv_kappa = inv_kappa

        self.lam_init = lam_init
        self.learnable_lam = learnable_lam

        if learnable_lam: self.lam = nn.Parameter(torch.empty(1))
        else: self.lam = nn.Parameter(torch.tensor(lam_init, requires_grad=False))
        
        if not off_gain:
            self.weight = nn.Parameter(torch.empty(hidden_size))
        else:
            # Use a constant tensor of ones for weight, with no gradient (not a parameter)
            self.register_buffer("weight", torch.ones(hidden_size), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.learnable_lam: self.lam.data.fill_(self.lam_init)
        self.weight.data.fill_(1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        var_x = (hidden_states * hidden_states).mean(dim=-1, keepdim=True, dtype=torch.float32)
        alpha = self.lam * torch.rsqrt(var_x + self.eps) * self.inv_kappa
        tanh_out = torch.tanh(alpha * hidden_states)
        if self.training:
            self._last_tanh_out = tanh_out.detach()
        return self.weight * tanh_out.to(dtype=hidden_states.dtype)


# ---- BHyT decode-time fused helpers ---------------------------------------
# Opt-in compile via env var BHYT_DECODE_COMPILE=1. These are pure functions
# extracted from _bhyt_decode_fastpath so torch.compile can fuse kernels
# without seeing the KV-cache self-attention path.

def _bhyt_decode_pre_norm(hidden_states, lam, weight_in, inv_kappa, var_dtype):
    """rstd1 + input-side BHyT norm. Returns (h_normed, var_x_fp32)."""
    var_x = (hidden_states * hidden_states).mean(dim=-1, keepdim=True, dtype=var_dtype)
    rstd1 = torch.rsqrt(var_x + 1e-6) * inv_kappa
    if rstd1.dtype != hidden_states.dtype:
        rstd1 = rstd1.to(dtype=hidden_states.dtype)
    tanh_out = torch.tanh(lam * rstd1 * hidden_states)
    return weight_in * tanh_out, var_x


def _bhyt_decode_post_norm(hidden_states, var_x, c_correction, lam, weight_post, inv_kappa):
    """rstd2 (baseline var-approx) + post-side BHyT norm.

    c_correction is precomputed: c_scaler * lam2_scaled  (× gamma_sq.pow(2).mean() if use_gamma_var).
    """
    var_x_next = var_x + c_correction
    rstd2 = torch.rsqrt(var_x_next + 1e-6) * inv_kappa
    if rstd2.dtype != hidden_states.dtype:
        rstd2 = rstd2.to(dtype=hidden_states.dtype)
    tanh_out = torch.tanh(lam * rstd2 * hidden_states)
    return weight_post * tanh_out


def _bhyt_decode_pre_norm_fast(hidden_states, lam_eff, weight_in):
    """Pre-norm using a precomputed scalar lam_eff = (lam · 1/κ) in hidden_states dtype.

    Caller is responsible for ensuring var is computed in hidden_states dtype (no cast).
    Saves: 1× inv_kappa scalar mul, 1× dtype cast — exact math, fewer ops in eager mode.
    """
    var_x = (hidden_states * hidden_states).mean(dim=-1, keepdim=True)
    rsqrt_v = torch.rsqrt(var_x + 1e-6)
    tanh_out = torch.tanh(lam_eff * rsqrt_v * hidden_states)
    return weight_in * tanh_out, var_x


def _bhyt_decode_post_norm_fast(hidden_states, var_x, c_correction_input, lam_eff, weight_post):
    """Post-norm using precomputed scalars (lam_eff, c_correction_input) in input dtype.

    var_x and c_correction_input must already be in hidden_states dtype.
    Saves: 1× inv_kappa scalar mul, 1× dtype cast vs the baseline post-norm.
    """
    var_x_next = var_x + c_correction_input
    rsqrt_v = torch.rsqrt(var_x_next + 1e-6)
    tanh_out = torch.tanh(lam_eff * rsqrt_v * hidden_states)
    return weight_post * tanh_out


_BHYT_DECODE_STREAM = os.environ.get("BHYT_DECODE_STREAM", "0") == "1"

if os.environ.get("BHYT_DECODE_COMPILE", "0") == "1":
    _bhyt_decode_pre_norm = torch.compile(
        _bhyt_decode_pre_norm, dynamic=False, fullgraph=False
    )
    _bhyt_decode_post_norm = torch.compile(
        _bhyt_decode_post_norm, dynamic=False, fullgraph=False
    )

# Opt-in Triton fused kernels (decode-only). Activated by BHYT_TRITON_KERNEL=1.
# Override the python helpers with thin shims that delegate to Triton on CUDA.
if os.environ.get("BHYT_TRITON_KERNEL", "0") == "1":
    try:
        from .bhyt_triton import bhyt_pre_norm_triton, bhyt_post_norm_triton

        def _bhyt_decode_pre_norm(hidden_states, lam, weight_in, inv_kappa, var_dtype):  # noqa: F811
            return bhyt_pre_norm_triton(hidden_states, lam, weight_in, inv_kappa)

        def _bhyt_decode_post_norm(hidden_states, var_x, c_correction, lam, weight_post, inv_kappa):  # noqa: F811
            return bhyt_post_norm_triton(
                hidden_states, var_x, c_correction, lam, weight_post, inv_kappa
            )
    except Exception as _e:  # pragma: no cover
        import warnings as _warnings
        _warnings.warn(f"BHYT_TRITON_KERNEL=1 requested but import failed: {_e}")


class BHyT(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, lam_init: float = 2.0, learnable_lam: bool = False, off_gain: bool = False):
        super().__init__()
        self.eps = eps

        self.lam_init = lam_init
        self.alpha = lam_init
        self.learnable_lam = learnable_lam

        if learnable_lam: self.lam = nn.Parameter(torch.empty(1))
        else: self.lam = nn.Parameter(torch.tensor(lam_init, requires_grad=False))
        
        if not off_gain:
            self.weight = nn.Parameter(torch.empty(hidden_size))
        else:
            # Use a constant tensor of ones for weight, with no gradient (not a parameter)
            self.register_buffer("weight", torch.ones(hidden_size), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.learnable_lam: self.lam.data.fill_(self.lam_init)
        self.weight.data.fill_(1)

    def forward(self, hidden_states: torch.Tensor, rstd: torch.Tensor) -> torch.Tensor:
        tanh_out = torch.tanh(self.lam * rstd * hidden_states)
        if self.training:
            self._last_tanh_out = tanh_out.detach()
        # norm_fold: when gamma is pre-folded into the next Linear, return without weight mul.
        if getattr(self, "_bhyt_fold_active", False):
            return tanh_out
        return self.weight * tanh_out

class LayerWiseVarApprox(nn.Module):
    def __init__(self, inv_kappa: float = 0.1):
        super().__init__()
        self.inv_kappa = inv_kappa
        self.inv_kappa_sq = inv_kappa ** 2
        self.calibration_scale = 1.0  # post-hoc per-layer var-approx correction (1.0 = no-op)

    def forward(self, x: torch.Tensor, c_scaler: float, lam: float, var_dtype: torch.dtype = torch.float32,
                var_approx_method: str = "baseline", var_beta=None, var_eta=None, gamma_sq=None,
                var_alpha=None, var_cross_beta=None):
        var_x = (x * x).mean(dim=-1, keepdim=True, dtype=var_dtype)
        rstd1 = torch.rsqrt(var_x + 1e-6) * self.inv_kappa

        if var_approx_method == "eta" and var_eta is not None:
            var_x_next = var_x * (1.0 + var_eta.to(var_dtype))
        else:
            c_scaler = c_scaler.to(var_dtype)
            lam2_scaled = lam.to(var_dtype).square() * self.inv_kappa_sq
            if gamma_sq is not None:
                lam2_scaled = lam2_scaled * gamma_sq.to(var_dtype)
            if var_approx_method == "residual_aware" and var_alpha is not None:
                var_x_next = var_x * (1.0 + var_cross_beta.to(var_dtype)) + var_alpha.to(var_dtype) * (c_scaler * lam2_scaled)
            elif var_approx_method == "beta" and var_beta is not None:
                var_x_next = var_x + var_beta.to(var_dtype) * (c_scaler * lam2_scaled)
            else:
                var_x_next = var_x + (c_scaler * lam2_scaled)

        if self.calibration_scale != 1.0:
            var_x_next = var_x_next * self.calibration_scale

        rstd2 = torch.rsqrt(var_x_next + 1e-6) * self.inv_kappa
        return rstd1.to(dtype=x.dtype), rstd2.to(dtype=x.dtype)

class LayerWiseVarApproxParrallel(nn.Module):
    def __init__(self, inv_kappa: float = 0.1):
        super().__init__()
        self.inv_kappa = inv_kappa
        self.inv_kappa_sq = inv_kappa ** 2
        self.calibration_scale = 1.0  # post-hoc per-layer var-approx correction (1.0 = no-op)

    def compute_rstd1(self, x: torch.Tensor, var_dtype: torch.dtype = torch.float32):
        var_x = (x * x).mean(dim=-1, keepdim=True, dtype=var_dtype)
        rstd1 = torch.rsqrt(var_x + 1e-6) * self.inv_kappa
        return rstd1.to(dtype=x.dtype), var_x  # return var_x explicitly

    def compute_rstd2(self, var_x: torch.Tensor, c_scaler: torch.Tensor, lam: torch.Tensor, var_dtype: torch.dtype = torch.float32,
                      var_approx_method: str = "baseline", var_beta=None, var_eta=None, gamma_sq=None,
                      var_alpha=None, var_cross_beta=None):
        hidden_states_dtype = lam.dtype

        if var_approx_method == "eta" and var_eta is not None:
            var_x_next = var_x * (1.0 + var_eta.to(var_dtype))
        else:
            c_scaler = c_scaler.to(var_dtype)
            lam2_scaled = lam.square().to(var_dtype) * self.inv_kappa_sq
            if gamma_sq is not None:
                lam2_scaled = lam2_scaled * gamma_sq.to(var_dtype)
            if var_approx_method == "residual_aware" and var_alpha is not None:
                var_x_next = var_x * (1.0 + var_cross_beta.to(var_dtype)) + var_alpha.to(var_dtype) * (c_scaler * lam2_scaled)
            elif var_approx_method == "beta" and var_beta is not None:
                var_x_next = var_x + var_beta.to(var_dtype) * (c_scaler * lam2_scaled)
            else:
                var_x_next = var_x + (c_scaler * lam2_scaled)

        if self.calibration_scale != 1.0:
            var_x_next = var_x_next * self.calibration_scale

        rstd2 = torch.rsqrt(var_x_next + 1e-6) * self.inv_kappa
        return rstd2.to(dtype=hidden_states_dtype)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # norm_fold: when gamma is pre-folded into the next Linear, return without weight mul.
        if getattr(self, "_bhyt_fold_active", False):
            return hidden_states.to(input_dtype)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)
ALL_LAYERNORM_LAYERS.append(BHyT)
ALL_LAYERNORM_LAYERS.append(BHyT_Star)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)

        bhyt_layer_range = getattr(config, 'bhyt_layer_range', None)
        if bhyt_layer_range is not None and config.norm_type in ["bhyt", "bhytline"]:
            if bhyt_layer_range[0] <= layer_idx <= bhyt_layer_range[1]:
                norm_type = config.norm_type
            else:
                norm_type = getattr(config, 'fallback_norm_type', 'rms')
        else:
            norm_type = config.norm_type
        self.norm_type = norm_type
        self.seq_len = config.seq_len
        self.layer_index = layer_idx
        self.rstd2_stream = None
        self.rstd1_done_event = None
        self.inference_var_dtype = getattr(config, "inference_var_dtype", "float32")
        self.var_approx_method = getattr(config, "var_approx_method", "baseline")
        self.use_gamma_var = getattr(config, "use_gamma_var", False)
        _base_kappa = getattr(config, "bhyt_kappa", 10.0)
        _kappa_schedule = getattr(config, "bhyt_kappa_schedule", None)
        if _kappa_schedule is not None:
            l = layer_idx + 1
            if _kappa_schedule == "decay":
                _kappa = _base_kappa / math.sqrt(l)
            elif _kappa_schedule == "grow":
                _kappa = _base_kappa * math.sqrt(l)
            elif _kappa_schedule == "mild_grow":
                _kappa = _base_kappa * (l ** 0.25)
            else:
                _kappa = _base_kappa
        else:
            _kappa = _base_kappa
        self.inv_kappa = 1.0 / _kappa
        self.inv_kappa_sq = self.inv_kappa ** 2
        self.var_approx_exact_layers = getattr(config, "var_approx_exact_layers", 0)
        _exact_list_raw = getattr(config, "var_approx_exact_layer_list", None)
        _exact_list = set()
        if _exact_list_raw is not None and str(_exact_list_raw).strip() != "":
            _exact_list = {int(x) for x in str(_exact_list_raw).split(",") if x.strip() != ""}
        self._use_exact_var = (
            (self.var_approx_exact_layers > 0 and layer_idx < self.var_approx_exact_layers)
            or (layer_idx in _exact_list)
        )
        self._var_aux_loss_active = False
        self._var_aux_loss = None

        if norm_type == "rms":
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif norm_type in ["bhyt", "bhytline"]:
            self.register_buffer("c_attn", torch.tensor(0.0), persistent=False)
            self._c_attn_ready = False

            if norm_type == 'bhytt':
                off_gain = True
            else: off_gain = False

            scale_factor = 1
            self.input_layernorm = BHyT(config.hidden_size, lam_init=config.input_layer_lam * scale_factor, learnable_lam=config.learnable_lam, off_gain=off_gain)
            self.post_attention_layernorm = BHyT(config.hidden_size, lam_init=config.post_layer_lam * scale_factor, learnable_lam=config.learnable_lam, off_gain=off_gain)

            if self.var_approx_method == "beta":
                self.var_beta = nn.Parameter(torch.tensor(1.0))
            elif self.var_approx_method == "eta":
                self.var_eta = nn.Parameter(torch.tensor(0.0))
            elif self.var_approx_method == "residual_aware":
                self.var_alpha = nn.Parameter(torch.tensor(1.0))
                self.var_cross_beta = nn.Parameter(torch.tensor(0.0))

        elif norm_type in ["bhytstar"]:
            scale_factor = 1
            off_gain = False
            self.input_layernorm = BHyT_Star(config.hidden_size, lam_init=config.input_layer_lam * scale_factor, learnable_lam=config.learnable_lam, off_gain=off_gain, inv_kappa=self.inv_kappa)
            self.post_attention_layernorm = BHyT_Star(config.hidden_size, lam_init=config.post_layer_lam * scale_factor, learnable_lam=config.learnable_lam, off_gain=off_gain, inv_kappa=self.inv_kappa)

        if norm_type in ["bhyt"]:
            self.var_approx = LayerWiseVarApproxParrallel(inv_kappa=self.inv_kappa)
        elif norm_type in ["bhytline"]:
            self.var_approx = LayerWiseVarApprox(inv_kappa=self.inv_kappa)

        # Optional post-hoc per-layer calibration scale on var_x_next (no-op when absent).
        _calib = getattr(config, 'var_approx_calibration', None)
        if _calib is not None and hasattr(self, 'var_approx') and layer_idx < len(_calib):
            self.var_approx.calibration_scale = float(_calib[layer_idx])

        # Post-hook: reinitialize var_eta/var_beta/var_alpha/var_cross_beta if missing from checkpoint (meta device safety)
        if hasattr(self, 'var_eta') or hasattr(self, 'var_beta') or hasattr(self, 'var_alpha'):
            self._register_load_state_dict_pre_hook(self._fix_missing_var_params, with_module=True)

    @staticmethod
    def _fix_missing_var_params(module, state_dict, prefix, *args, **kwargs):
        eta_key = prefix + 'var_eta'
        beta_key = prefix + 'var_beta'
        alpha_key = prefix + 'var_alpha'
        cross_beta_key = prefix + 'var_cross_beta'
        if hasattr(module, 'var_eta') and eta_key not in state_dict:
            state_dict[eta_key] = torch.tensor(0.0, dtype=module.var_eta.dtype, device=module.var_eta.device)
        if hasattr(module, 'var_beta') and beta_key not in state_dict:
            state_dict[beta_key] = torch.tensor(1.0, dtype=module.var_beta.dtype, device=module.var_beta.device)
        if hasattr(module, 'var_alpha') and alpha_key not in state_dict:
            state_dict[alpha_key] = torch.tensor(1.0, dtype=module.var_alpha.dtype, device=module.var_alpha.device)
        if hasattr(module, 'var_cross_beta') and cross_beta_key not in state_dict:
            state_dict[cross_beta_key] = torch.tensor(0.0, dtype=module.var_cross_beta.dtype, device=module.var_cross_beta.device)

    @torch.no_grad()
    def update_scaler(self):
        W_in_V   = self.self_attn.v_proj.weight.detach().to(dtype=torch.float32)  # [dv_compact, d]
        W_out    = self.self_attn.o_proj.weight.detach().to(dtype=torch.float32)  # [d, d_full]
        if W_in_V.is_meta or W_out.is_meta: return False

        # GQA: expand V weights to match repeat_kv before computing combined matrix
        num_kv_groups = getattr(self.self_attn, 'num_key_value_groups', 1)
        if num_kv_groups > 1:
            head_dim = self.self_attn.head_dim
            num_kv_heads = W_in_V.shape[0] // head_dim
            W_V_expanded = W_in_V.reshape(num_kv_heads, head_dim, -1)
            W_V_expanded = W_V_expanded.repeat_interleave(num_kv_groups, dim=0)
            W_V_expanded = W_V_expanded.reshape(-1, W_in_V.shape[1])  # [d_full, d]
        else:
            W_V_expanded = W_in_V  # MHA: no expansion needed

        # Paper formula: ||W_V^paper @ W_O^paper||_F^2
        # W_V^paper = W_V_expanded.T, W_O^paper = W_out.T
        W_combined = torch.matmul(W_V_expanded.T, W_out.T)  # [d, d]
        w_vo = torch.norm(W_combined, p='fro') ** 2

        denom = float(self.seq_len * self.hidden_size)
        c_tensor = (w_vo / denom).detach()

        if torch.compiler.is_compiling():
            # stay in tensor-land to avoid .item() during graph capture
            new_c_attn = c_tensor.clone()
        else:
            # eager path can afford to materialize a Python float
            new_c_attn = torch.tensor(float(c_tensor), device=W_in_V.device, dtype=W_in_V.dtype)

        new_c_attn = new_c_attn.to(device=W_in_V.device, dtype=W_in_V.dtype)

        if isinstance(getattr(self, "c_attn", None), torch.Tensor) and not self.c_attn.is_meta:
            self.c_attn.copy_(new_c_attn)
        else:
            self.c_attn = new_c_attn

        self._c_attn_ready = True
        return True

    @torch.no_grad()
    def precompute_bhyt_constants(self, target_dtype: torch.dtype) -> bool:
        """Pre-compute scalar constants (lam · 1/κ, c_correction) in `target_dtype`.

        Activates the fast-path `_bhyt_decode_pre_norm_fast` / `_bhyt_decode_post_norm_fast`.
        Saves: 2× scalar mul + 2× dtype cast per layer (pre + post). Math is exact.
        Caller must also set inference_var_dtype='input' so var is computed in target_dtype.
        BHyT layers only — silently no-ops for non-BHyT (RMS) layers.
        """
        if not all(hasattr(self, name) for name in ("input_layernorm", "post_attention_layernorm")):
            return False
        if not (hasattr(self.input_layernorm, "lam") and hasattr(self.post_attention_layernorm, "lam")):
            return False
        if getattr(self, "c_attn", None) is None or self.c_attn.is_meta:
            return False
        lam_in = self.input_layernorm.lam.detach()
        lam_post = self.post_attention_layernorm.lam.detach()
        self._lam_eff_pre = (lam_in * self.inv_kappa).to(target_dtype)
        self._lam_eff_post = (lam_post * self.inv_kappa).to(target_dtype)
        # c_correction in target dtype (BHyT post-norm var approximation constant)
        lam2_scaled = (lam_in * lam_in).to(target_dtype) * self.inv_kappa_sq
        c_scaler = self.c_attn.to(target_dtype)
        self._c_correction_fast = (c_scaler * lam2_scaled).to(target_dtype)
        self._precompute_active = True
        return True

    def _get_var_dtype(self, hidden_states: torch.Tensor) -> torch.dtype:
        if self.training:
            return torch.float32
        if self.inference_var_dtype == "input" and hidden_states.dtype in (torch.float16, torch.bfloat16):
            return hidden_states.dtype
        return torch.float32

    def _bhyt_decode_fastpath(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_value: Optional[Cache],
        output_attentions: Optional[bool],
        use_cache: Optional[bool],
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states

        # Fused pre-norm chunk (rstd1 + input BHyT norm) when var_approx is baseline.
        # `None` is the historical default that means baseline (config may omit the field).
        # The fused helper is module-level so torch.compile can be applied via env var.
        # norm_fold: when gamma is folded into the downstream Linear, the fused helpers
        # cannot be used (they apply self.weight inline). Fall back to the eager path,
        # which routes through self.input_layernorm / self.post_attention_layernorm —
        # both of which now honor the _bhyt_fold_active gate and skip the gamma mul.
        _fold_active = getattr(self.input_layernorm, "_bhyt_fold_active", False) or \
                       getattr(self.post_attention_layernorm, "_bhyt_fold_active", False)
        _use_fused = (
            self.var_approx_method in (None, "baseline")
            and not self.use_gamma_var
            and not self._use_exact_var
            and not _fold_active
        )
        # Generation precompute fast path (default-ON, decode-only). Lazily precompute
        # the exact scalars (lam·1/κ, c_correction) in the input dtype the first time
        # we decode, so the fast path engages without any opt-in flag. This is scoped
        # to the decode helper, so prefill/eval variance dtype is left unchanged. The
        # transform is mathematically exact (associativity) — greedy decoding is identical.
        if (_use_fused
                and not self.training
                and hidden_states.dtype in (torch.float16, torch.bfloat16)
                and (not getattr(self, "_precompute_active", False)
                     or self._lam_eff_pre.dtype != hidden_states.dtype)):
            self.precompute_bhyt_constants(hidden_states.dtype)

        var_dtype = self._get_var_dtype(hidden_states)
        # When precompute is active, compute variance in the input dtype so it matches
        # the precomputed scalars and the exact fast path is taken (decode-only override).
        if (_use_fused
                and getattr(self, "_precompute_active", False)
                and self._lam_eff_pre.dtype == hidden_states.dtype):
            var_dtype = hidden_states.dtype

        _use_precomputed = (
            _use_fused
            and getattr(self, "_precompute_active", False)
            and var_dtype == hidden_states.dtype
        )
        if _use_precomputed:
            hidden_states, var_x = _bhyt_decode_pre_norm_fast(
                hidden_states,
                self._lam_eff_pre,
                self.input_layernorm.weight,
            )
        elif _use_fused:
            hidden_states, var_x = _bhyt_decode_pre_norm(
                hidden_states,
                self.input_layernorm.lam,
                self.input_layernorm.weight,
                self.inv_kappa,
                var_dtype,
            )
        else:
            var_x = (hidden_states * hidden_states).mean(
                dim=-1,
                keepdim=True,
                dtype=var_dtype,
            )
            rstd1 = torch.rsqrt(var_x + 1e-6) * self.inv_kappa
            if rstd1.dtype != hidden_states.dtype:
                rstd1 = rstd1.to(dtype=hidden_states.dtype)
            hidden_states = self.input_layernorm(hidden_states, rstd1)

        # === Opt-in stream-parallel rstd2 (overlaps with self_attn) =========
        # Activated by BHYT_DECODE_STREAM=1. Only safe in fused baseline path.
        _stream_rstd2 = None
        if (_BHYT_DECODE_STREAM
                and _use_fused
                and not self.training
                and hidden_states.is_cuda):
            # Cache c_correction
            if getattr(self, "_bhyt_cached_c_correction_dtype", None) != var_dtype:
                lam_d = self.input_layernorm.lam.detach()
                lam2_scaled = (lam_d * lam_d).to(var_dtype) * self.inv_kappa_sq
                c_scaler = self.c_attn.to(var_dtype)
                self._bhyt_cached_c_correction = c_scaler * lam2_scaled
                self._bhyt_cached_c_correction_dtype = var_dtype
            # Lazy-init stream + event
            if self.rstd2_stream is None:
                self.rstd2_stream = torch.cuda.Stream(device=hidden_states.device)
                self.rstd1_done_event = torch.cuda.Event()
            # Side stream computes rstd2 concurrent with main-stream self_attn
            torch.cuda.current_stream().record_event(self.rstd1_done_event)
            _c_corr = self._bhyt_cached_c_correction
            with torch.cuda.stream(self.rstd2_stream):
                self.rstd2_stream.wait_event(self.rstd1_done_event)
                _var_x_next = var_x + _c_corr
                _r = torch.rsqrt(_var_x_next + 1e-6) * self.inv_kappa
                if _r.dtype != hidden_states.dtype:
                    _r = _r.to(dtype=hidden_states.dtype)
                _stream_rstd2 = _r

        # 2) Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        # Stream-parallel path: rstd2 already on side stream → sync & apply BHyT
        if _stream_rstd2 is not None:
            torch.cuda.current_stream().wait_stream(self.rstd2_stream)
            hidden_states = self.post_attention_layernorm(hidden_states, _stream_rstd2)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states, self_attn_weights

        # Inference: c_correction = c_attn * lam² / κ² is constant across all
        # tokens for a given dtype, so compute once per layer + var_dtype and
        # reuse. Training still recomputes since `lam` may update.
        # cudagraph compatibility: when `_disable_cache=True` is set on the layer,
        # recompute every step (no attribute write on the hot path). PyTorch eager
        # fuses these scalar ops anyway — cache effect is ≈0 (report §4.3).
        if _use_precomputed and not self.training:
            hidden_states = _bhyt_decode_post_norm_fast(
                hidden_states,
                var_x,
                self._c_correction_fast,
                self._lam_eff_post,
                self.post_attention_layernorm.weight,
            )
        elif _use_fused and not self.training:
            if getattr(self, "_disable_cache", False):
                lam_d = self.input_layernorm.lam.detach()
                lam2_scaled = (lam_d * lam_d).to(var_dtype) * self.inv_kappa_sq
                c_scaler = self.c_attn.to(var_dtype)
                _c_corr_local = c_scaler * lam2_scaled
            else:
                if getattr(self, "_bhyt_cached_c_correction_dtype", None) != var_dtype:
                    lam_d = self.input_layernorm.lam.detach()
                    lam2_scaled = (lam_d * lam_d).to(var_dtype) * self.inv_kappa_sq
                    c_scaler = self.c_attn.to(var_dtype)
                    self._bhyt_cached_c_correction = c_scaler * lam2_scaled
                    self._bhyt_cached_c_correction_dtype = var_dtype
                _c_corr_local = self._bhyt_cached_c_correction
            hidden_states = _bhyt_decode_post_norm(
                hidden_states,
                var_x,
                _c_corr_local,
                self.post_attention_layernorm.lam,
                self.post_attention_layernorm.weight,
                self.inv_kappa,
            )
        elif _use_fused:
            # Training fused path: recompute every step.
            lam_d = self.input_layernorm.lam.detach()
            lam2_scaled = (lam_d * lam_d).to(var_dtype) * self.inv_kappa_sq
            c_scaler = self.c_attn.to(var_dtype)
            c_correction = c_scaler * lam2_scaled
            hidden_states = _bhyt_decode_post_norm(
                hidden_states,
                var_x,
                c_correction,
                self.post_attention_layernorm.lam,
                self.post_attention_layernorm.weight,
                self.inv_kappa,
            )
        else:
            # Non-fused branch needs c_scaler & lam2_scaled separately for the
            # eta / residual_aware / beta variants and gamma_var multiplication.
            if not self.training and getattr(self, "_bhyt_cached_lam2_scaled_dtype", None) == var_dtype:
                lam2_scaled = self._bhyt_cached_lam2_scaled
                c_scaler = self._bhyt_cached_c_attn_full
            else:
                lam_d = self.input_layernorm.lam.detach()
                lam2_scaled = (lam_d * lam_d).to(var_dtype) * self.inv_kappa_sq
                c_scaler = self.c_attn.to(var_dtype)
                if not self.training:
                    self._bhyt_cached_lam2_scaled = lam2_scaled
                    self._bhyt_cached_c_attn_full = c_scaler
                    self._bhyt_cached_lam2_scaled_dtype = var_dtype

            if self.var_approx_method == "eta" and hasattr(self, "var_eta"):
                var_x_next = var_x * (1.0 + self.var_eta.to(var_dtype))
            elif self.var_approx_method == "residual_aware" and hasattr(self, "var_alpha"):
                _correction = c_scaler * lam2_scaled
                if self.use_gamma_var:
                    _correction = _correction * self.input_layernorm.weight.detach().pow(2).mean().to(var_dtype)
                var_x_next = var_x * (1.0 + self.var_cross_beta.to(var_dtype)) + self.var_alpha.to(var_dtype) * _correction
            elif self.var_approx_method == "beta" and hasattr(self, "var_beta"):
                _correction = c_scaler * lam2_scaled
                if self.use_gamma_var:
                    _correction = _correction * self.input_layernorm.weight.detach().pow(2).mean().to(var_dtype)
                var_x_next = var_x + self.var_beta.to(var_dtype) * _correction
            else:
                _correction = c_scaler * lam2_scaled
                if self.use_gamma_var:
                    _correction = _correction * self.input_layernorm.weight.detach().pow(2).mean().to(var_dtype)
                var_x_next = var_x + _correction

            # Hybrid: use exact variance for early layers
            if self._use_exact_var:
                exact_var = (hidden_states * hidden_states).mean(dim=-1, keepdim=True, dtype=var_dtype)
                rstd2 = torch.rsqrt(exact_var + 1e-6) * self.inv_kappa
            else:
                rstd2 = torch.rsqrt(var_x_next + 1e-6) * self.inv_kappa
            if rstd2.dtype != hidden_states.dtype:
                rstd2 = rstd2.to(dtype=hidden_states.dtype)

            hidden_states = self.post_attention_layernorm(hidden_states, rstd2)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, self_attn_weights
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        if self.norm_type in ["bhytline"]:
            if (not self._c_attn_ready) or (isinstance(self.c_attn, torch.Tensor) and self.c_attn.is_meta):
                self.seq_len = hidden_states.shape[1]
                self.update_scaler()
            var_dtype = self._get_var_dtype(hidden_states)
            residual = hidden_states
            _gamma_sq = self.input_layernorm.weight.detach().pow(2).mean() if self.use_gamma_var else None
            rstd1, rstd2 = self.var_approx(hidden_states, self.c_attn, self.input_layernorm.lam.detach(), var_dtype=var_dtype,
                                           var_approx_method=self.var_approx_method,
                                           var_beta=getattr(self, 'var_beta', None),
                                           var_eta=getattr(self, 'var_eta', None),
                                           gamma_sq=_gamma_sq,
                                           var_alpha=getattr(self, 'var_alpha', None),
                                           var_cross_beta=getattr(self, 'var_cross_beta', None))
            hidden_states = self.input_layernorm(hidden_states, rstd1)

            # Self Attention
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states

            # Hybrid: override rstd2 with exact variance for early layers
            if self._use_exact_var:
                exact_var = (hidden_states * hidden_states).mean(dim=-1, keepdim=True, dtype=var_dtype)
                rstd2 = (torch.rsqrt(exact_var + 1e-6) * self.inv_kappa).to(dtype=hidden_states.dtype)

            # Collect variance approximation data when flag is set
            if getattr(self, '_collect_var_approx', False):
                with torch.no_grad():
                    actual_var = (hidden_states * hidden_states).mean(dim=-1, keepdim=True, dtype=var_dtype)
                    approx_var = (self.inv_kappa / rstd2.to(var_dtype)).square() - 1e-6
                    self._var_approx_data["actual"].append(actual_var.detach().cpu())
                    self._var_approx_data["approx"].append(approx_var.detach().cpu())

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states, rstd2)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        elif self.norm_type in ["bhyt"]:
            # if (not self._c_attn_ready) or (isinstance(self.c_attn, torch.Tensor) and self.c_attn.is_meta):
            if not self._c_attn_ready:
                self.seq_len = hidden_states.shape[1]
                self.update_scaler()
            is_decoding = hidden_states.shape[1] == 1
            var_dtype = self._get_var_dtype(hidden_states)

            if is_decoding:
                hidden_states, self_attn_weights = self._bhyt_decode_fastpath(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    **kwargs,
                )
            else:
                residual = hidden_states
                rstd1, var_x = self.var_approx.compute_rstd1(hidden_states, var_dtype=var_dtype)
                hidden_states = self.input_layernorm(hidden_states, rstd1)

                rstd2 = None
                use_cuda_parallel = hidden_states.is_cuda
                if use_cuda_parallel:
                    if self.rstd2_stream is None:
                        self.rstd2_stream = torch.cuda.Stream(device=hidden_states.device)
                        self.rstd1_done_event = torch.cuda.Event()
                    rstd2_stream = self.rstd2_stream
                    rstd1_done = self.rstd1_done_event
                    torch.cuda.current_stream().record_event(rstd1_done)

                    lam_detached = self.input_layernorm.lam.detach()
                    c_scaler = self.c_attn
                    _gamma_sq = self.input_layernorm.weight.detach().pow(2).mean() if self.use_gamma_var else None

                    _var_beta = getattr(self, 'var_beta', None)
                    _var_eta = getattr(self, 'var_eta', None)
                    _var_alpha = getattr(self, 'var_alpha', None)
                    _var_cross_beta = getattr(self, 'var_cross_beta', None)
                    _var_method = self.var_approx_method

                    with torch.cuda.stream(rstd2_stream):
                        rstd2_stream.wait_event(rstd1_done)
                        rstd2 = self.var_approx.compute_rstd2(var_x, c_scaler, lam_detached, var_dtype=var_dtype,
                                                              var_approx_method=_var_method, var_beta=_var_beta, var_eta=_var_eta, gamma_sq=_gamma_sq,
                                                              var_alpha=_var_alpha, var_cross_beta=_var_cross_beta)
                else:
                    lam_detached = self.input_layernorm.lam.detach()
                    c_scaler = self.c_attn

                # Self Attention
                hidden_states, self_attn_weights = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states

                if use_cuda_parallel:
                    torch.cuda.current_stream().wait_stream(rstd2_stream)
                else:
                    _gamma_sq = self.input_layernorm.weight.detach().pow(2).mean() if self.use_gamma_var else None
                    rstd2 = self.var_approx.compute_rstd2(var_x, c_scaler, lam_detached, var_dtype=var_dtype,
                                                          var_approx_method=self.var_approx_method,
                                                          var_beta=getattr(self, 'var_beta', None),
                                                          var_eta=getattr(self, 'var_eta', None),
                                                          gamma_sq=_gamma_sq,
                                                          var_alpha=getattr(self, 'var_alpha', None),
                                                          var_cross_beta=getattr(self, 'var_cross_beta', None))

                # Hybrid: override rstd2 with exact variance for early layers
                if self._use_exact_var:
                    exact_var = (hidden_states * hidden_states).mean(dim=-1, keepdim=True, dtype=var_dtype)
                    rstd2 = (torch.rsqrt(exact_var + 1e-6) * self.inv_kappa).to(dtype=hidden_states.dtype)

                # Collect variance approximation data when flag is set
                if getattr(self, '_collect_var_approx', False):
                    with torch.no_grad():
                        actual_var = (hidden_states * hidden_states).mean(dim=-1, keepdim=True, dtype=var_dtype)
                        approx_var = (self.inv_kappa / rstd2.to(var_dtype)).square() - 1e-6
                        self._var_approx_data["actual"].append(actual_var.detach().cpu())
                        self._var_approx_data["approx"].append(approx_var.detach().cpu())

                # Variance approximation auxiliary loss
                if self._var_aux_loss_active and not self._use_exact_var:
                    actual_var = (hidden_states * hidden_states).mean(dim=-1, keepdim=True, dtype=var_dtype)
                    approx_var = (self.inv_kappa / rstd2.to(var_dtype)).square() - 1e-6
                    log_actual = torch.log(actual_var.detach() + 1e-8)
                    log_approx = torch.log(approx_var + 1e-8)
                    self._var_aux_loss = (log_approx - log_actual).square().mean()

                hidden_states = self.post_attention_layernorm(hidden_states, rstd2)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

            hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # BHyT variance approximation parameters
        if hasattr(module, 'var_eta') and isinstance(module.var_eta, nn.Parameter):
            module.var_eta.data.zero_()
        if hasattr(module, 'var_beta') and isinstance(module.var_beta, nn.Parameter):
            module.var_beta.data.fill_(1.0)


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        norm_type = config.norm_type
        self.norm_type = norm_type

        if (self.norm_type in ['bhyt', 'bhytline']) and config.last_layer_lam > 0.:
            if self.norm_type == 'bhyt': off_gain = False
            else: off_gain = False
            self.norm = BHyT(config.hidden_size, lam_init=config.last_layer_lam, learnable_lam=True, eps=config.rms_norm_eps, off_gain=off_gain)
            self.use_rmsnorm = False
        elif self.norm_type == ['bhytstar'] and config.last_layer_lam > 0.:
            off_gain = False
            _base_kappa_last = getattr(config, 'bhyt_kappa', 10.0)
            _kappa_sched_last = getattr(config, 'bhyt_kappa_schedule', None)
            if _kappa_sched_last is not None:
                _l = config.num_hidden_layers + 1
                if _kappa_sched_last == "decay":
                    _base_kappa_last = _base_kappa_last / math.sqrt(_l)
                elif _kappa_sched_last == "grow":
                    _base_kappa_last = _base_kappa_last * math.sqrt(_l)
                elif _kappa_sched_last == "mild_grow":
                    _base_kappa_last = _base_kappa_last * (_l ** 0.25)
            _inv_kappa = 1.0 / _base_kappa_last
            self.norm = BHyT_Star(config.hidden_size, lam_init=config.last_layer_lam, learnable_lam=True, eps=config.rms_norm_eps, off_gain=off_gain, inv_kappa=_inv_kappa)
            self.use_rmsnorm = False
        else:
            self.use_rmsnorm = True
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.norm_type in ['bhyt', 'bhytline']:
            self.precompute_c_attn(seq_len=config.seq_len)

        if getattr(self.config, 'use_shared_scale', False):
            self.shared_scale = nn.Parameter(torch.empty(1))
            self.shared_scale.data.fill_(math.sqrt(self.config.hidden_size))
        else:
            self.register_buffer("shared_scale", torch.ones(1, dtype=torch.float32), persistent=False)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def precompute_c_attn(self, seq_len: Optional[int] = None):
        """
        Precompute c_attn for BHyT-family norms to avoid on-the-fly heavy matmuls during the first decode token.
        """
        if self.norm_type not in ["bhyt", "bhytline"]:
            return
        seq_len = seq_len if seq_len is not None else getattr(self.config, "seq_len", None)
        if seq_len is None:
            return
        for layer in self.layers:
            if not hasattr(layer, "update_scaler"):
                continue
            if getattr(layer, "_c_attn_ready", True):
                continue
            layer.seq_len = seq_len
            layer.update_scaler()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        hidden_states = hidden_states * self.shared_scale.to(hidden_states.dtype)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        if (self.norm_type in ['bhyt', 'bhytline']) and not self.use_rmsnorm:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True, dtype=input_dtype)
            _base_kappa_fwd = getattr(self.config, 'bhyt_kappa', 10.0)
            _kappa_sched_fwd = getattr(self.config, 'bhyt_kappa_schedule', None)
            if _kappa_sched_fwd is not None:
                _l = self.config.num_hidden_layers + 1
                if _kappa_sched_fwd == "decay":
                    _base_kappa_fwd = _base_kappa_fwd / math.sqrt(_l)
                elif _kappa_sched_fwd == "grow":
                    _base_kappa_fwd = _base_kappa_fwd * math.sqrt(_l)
                elif _kappa_sched_fwd == "mild_grow":
                    _base_kappa_fwd = _base_kappa_fwd * (_l ** 0.25)
            _inv_kappa = 1.0 / _base_kappa_fwd
            rstd = _inv_kappa * torch.rsqrt(variance + 1e-6)
            hidden_states = self.norm(hidden_states.to(input_dtype), rstd)
        elif self.norm_type == 'bhytstar' and not self.use_rmsnorm:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # ---- norm_fold integration --------------------------------------------
    # If a user has applied gamma fold-in for inference speedup, switching
    # back to training mode must restore the original weights so that gradient
    # updates land on the unfolded parameters. Likewise, save_pretrained must
    # serialize unfolded weights so the checkpoint round-trips correctly.
    def train(self, mode: bool = True):
        if mode:
            try:
                from .norm_fold import is_folded, revert_norm_fold
            except ImportError:  # pragma: no cover — safety net
                return super().train(mode)
            if is_folded(self):
                revert_norm_fold(self)
        return super().train(mode)

    def save_pretrained(self, *args, **kwargs):
        try:
            from .norm_fold import apply_norm_fold, is_folded, revert_norm_fold
        except ImportError:  # pragma: no cover
            return super().save_pretrained(*args, **kwargs)
        was_folded = is_folded(self)
        if was_folded:
            revert_norm_fold(self)
        try:
            return super().save_pretrained(*args, **kwargs)
        finally:
            if was_folded:
                apply_norm_fold(self)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            # Add variance approximation auxiliary loss
            var_aux_weight = getattr(self.config, 'var_aux_loss_weight', 0.0)
            if var_aux_weight > 0:
                var_aux_losses = []
                for layer in self.model.layers:
                    if layer._var_aux_loss is not None:
                        var_aux_losses.append(layer._var_aux_loss)
                        layer._var_aux_loss = None
                if var_aux_losses:
                    loss = loss + var_aux_weight * torch.stack(var_aux_losses).mean()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]