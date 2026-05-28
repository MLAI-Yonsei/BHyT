"""gamma fold-in optimization (exact mathematical transformation).

Folds ``norm.weight`` (gamma) into the next Linear's weight for inference-only
speedup. By associativity of element-wise mul and matmul:

    W @ (gamma * u) == (W * gamma[None, :]) @ u

So we can pre-multiply gamma into the rows-input dimension of every Linear that
consumes the norm output, and skip the per-token element-wise multiplication.

Applies to BHyT / DyT / LlamaRMSNorm whose forward output is
``gamma * f(x)``. Other norms (BHyT_Star, BHyT_F1, Peri, SeeDNorm, LayerNorm)
are skipped — they either have additional dynamic per-token scaling or are
otherwise non-foldable.

Public API
==========
* :func:`can_fold(norm)`         — predicate for whitelist.
* :func:`apply_norm_fold(model)` — fold all eligible layers in-place. Reversible.
* :func:`revert_norm_fold(model)` — restore original linear weights from buffers.
* :func:`is_folded(model)`       — True if any decoder layer is in folded state.

Design notes
============
* Default OFF. Activation requires an explicit call to ``apply_norm_fold``.
* Reversible: the original Linear weights are kept as buffers on each Linear
  (``_bhyt_norm_fold_backup_<linear_name>``) so revert is bit-exact (no
  division-by-small-gamma instability).
* The fold multiplication is performed in fp32 then cast back to the original
  dtype to avoid fp16 round-off — tolerance is ~1e-3 for fp16/bf16, much
  tighter for fp32.
* The corresponding norm modules have an ``_bhyt_fold_active`` flag set; the
  forward gates in modeling_llama.py skip the ``gamma * tanh_out`` mul when
  this flag is True.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Class names that are eligible for fold-in. Whitelist by class name to avoid
# importing the classes (which would create a circular import with
# modeling_llama.py).
_FOLDABLE_NORM_CLASSES = frozenset({"BHyT", "DyT", "LlamaRMSNorm"})

# Attribute name on each decoder layer set after fold is applied.
_LAYER_FOLDED_FLAG = "_bhyt_norm_folded"
# Attribute name on each norm module after fold is applied.
_NORM_FOLD_ACTIVE_FLAG = "_bhyt_fold_active"
# Buffer-name prefix on each Linear to back up the pre-fold weight.
_LINEAR_BACKUP_PREFIX = "_bhyt_norm_fold_backup_"


def can_fold(norm_module: nn.Module) -> bool:
    """Return True if ``norm_module`` is eligible for gamma fold-in."""
    return norm_module.__class__.__name__ in _FOLDABLE_NORM_CLASSES


def _iter_decoder_layers(model: nn.Module):
    """Yield all ``LlamaDecoderLayer`` instances within ``model``."""
    for module in model.modules():
        if module.__class__.__name__ == "LlamaDecoderLayer":
            yield module


def _get_fold_targets(layer: nn.Module) -> List[Tuple[nn.Module, List[Tuple[str, nn.Linear]]]]:
    """Return list of (norm, [(linear_name, linear_module), ...]) pairs for a layer.

    Two pairs are returned when both ``input_layernorm`` and
    ``post_attention_layernorm`` are present:

    * ``input_layernorm`` -> ``self_attn.q_proj``, ``k_proj``, ``v_proj``
    * ``post_attention_layernorm`` -> ``mlp.gate_proj``, ``up_proj``
    """
    pairs: List[Tuple[nn.Module, List[Tuple[str, nn.Linear]]]] = []
    norm_in = getattr(layer, "input_layernorm", None)
    self_attn = getattr(layer, "self_attn", None)
    if norm_in is not None and self_attn is not None:
        linears = []
        for name in ("q_proj", "k_proj", "v_proj"):
            lin = getattr(self_attn, name, None)
            if isinstance(lin, nn.Linear):
                linears.append((f"self_attn.{name}", lin))
        if linears:
            pairs.append((norm_in, linears))
    norm_post = getattr(layer, "post_attention_layernorm", None)
    mlp = getattr(layer, "mlp", None)
    if norm_post is not None and mlp is not None:
        linears = []
        for name in ("gate_proj", "up_proj"):
            lin = getattr(mlp, name, None)
            if isinstance(lin, nn.Linear):
                linears.append((f"mlp.{name}", lin))
        if linears:
            pairs.append((norm_post, linears))
    return pairs


def _fold_linear(linear: nn.Linear, gamma: torch.Tensor, backup_key: str) -> None:
    """Multiply ``linear.weight`` by ``gamma`` on the input dim, in fp32.

    Backs up the original weight as a buffer on ``linear`` so that revert is
    bit-exact.
    """
    if hasattr(linear, backup_key):
        # already folded — guard against double-apply.
        raise RuntimeError(
            f"Linear already has backup buffer '{backup_key}'. Did you call apply_norm_fold twice?"
        )
    orig_dtype = linear.weight.dtype
    # Persistent=False: this is a runtime-only transformation; we do not want
    # the backup buffer to be saved into the state_dict (otherwise it would
    # double the on-disk size).
    linear.register_buffer(backup_key, linear.weight.detach().clone(), persistent=False)
    # Compute fold in fp32 to minimize round-off error, cast back.
    new_weight = (linear.weight.data.float() * gamma.float()[None, :]).to(orig_dtype)
    linear.weight.data = new_weight


def _unfold_linear(linear: nn.Linear, backup_key: str) -> None:
    """Restore ``linear.weight`` from the buffer and remove the buffer."""
    if not hasattr(linear, backup_key):
        raise RuntimeError(
            f"Linear has no backup buffer '{backup_key}'. revert_norm_fold called without prior apply?"
        )
    backup: torch.Tensor = getattr(linear, backup_key)
    linear.weight.data.copy_(backup)
    # Remove the buffer cleanly.
    delattr(linear, backup_key)
    # nn.Module tracks buffers in self._buffers — delattr leaves that dict
    # stale; clean it up.
    if backup_key in linear._buffers:
        del linear._buffers[backup_key]


@torch.no_grad()
def apply_norm_fold(model: nn.Module, verbose: bool = False) -> int:
    """Fold gamma into the downstream Linear weights for every eligible layer.

    Steps per layer:
      1. For each foldable norm (``input_layernorm``, ``post_attention_layernorm``):
         a. For each consuming Linear: back up its weight, multiply by gamma.
         b. Set ``norm._bhyt_fold_active = True`` so forward skips ``gamma *``.
      2. Mark ``layer._bhyt_norm_folded = True``.

    Args:
        model: A LlamaForCausalLM (or any nn.Module containing LlamaDecoderLayers).
        verbose: If True, log every folded layer index and warnings about
            skipped non-foldable norms.

    Returns:
        Number of decoder layers that were folded.
    """
    folded_count = 0
    for layer_idx, layer in enumerate(_iter_decoder_layers(model)):
        if getattr(layer, _LAYER_FOLDED_FLAG, False):
            if verbose:
                logger.info(f"[norm_fold] layer {layer_idx} already folded — skip.")
            continue
        pairs = _get_fold_targets(layer)
        # Verify every norm in this layer is foldable. If any is not, skip the
        # whole layer (mixed-norm layers are not supported by this pass).
        all_foldable = all(can_fold(norm) for norm, _ in pairs)
        if not all_foldable or not pairs:
            if verbose:
                cls_names = [norm.__class__.__name__ for norm, _ in pairs]
                logger.warning(
                    f"[norm_fold] layer {layer_idx} has non-foldable norms {cls_names} — skip."
                )
            continue
        # Apply fold to every (norm, [linears]) pair.
        for norm, linears in pairs:
            gamma = norm.weight.detach()
            for lin_name, lin in linears:
                backup_key = _LINEAR_BACKUP_PREFIX + lin_name.replace(".", "_")
                _fold_linear(lin, gamma, backup_key)
            setattr(norm, _NORM_FOLD_ACTIVE_FLAG, True)
        setattr(layer, _LAYER_FOLDED_FLAG, True)
        folded_count += 1
        if verbose:
            logger.info(f"[norm_fold] folded layer {layer_idx}.")
    if verbose:
        logger.info(f"[norm_fold] folded {folded_count} layer(s) total.")
    return folded_count


@torch.no_grad()
def revert_norm_fold(model: nn.Module) -> int:
    """Restore the original Linear weights and clear all fold flags.

    Returns:
        Number of decoder layers that were reverted.
    """
    reverted_count = 0
    for layer_idx, layer in enumerate(_iter_decoder_layers(model)):
        if not getattr(layer, _LAYER_FOLDED_FLAG, False):
            continue
        pairs = _get_fold_targets(layer)
        for norm, linears in pairs:
            for lin_name, lin in linears:
                backup_key = _LINEAR_BACKUP_PREFIX + lin_name.replace(".", "_")
                if hasattr(lin, backup_key):
                    _unfold_linear(lin, backup_key)
            if hasattr(norm, _NORM_FOLD_ACTIVE_FLAG):
                # Use __delattr__ on the instance attribute.
                try:
                    delattr(norm, _NORM_FOLD_ACTIVE_FLAG)
                except AttributeError:
                    setattr(norm, _NORM_FOLD_ACTIVE_FLAG, False)
        try:
            delattr(layer, _LAYER_FOLDED_FLAG)
        except AttributeError:
            setattr(layer, _LAYER_FOLDED_FLAG, False)
        reverted_count += 1
    return reverted_count


def is_folded(model: nn.Module) -> bool:
    """Return True if any decoder layer in ``model`` is currently in folded state."""
    for layer in _iter_decoder_layers(model):
        if getattr(layer, _LAYER_FOLDED_FLAG, False):
            return True
    return False
