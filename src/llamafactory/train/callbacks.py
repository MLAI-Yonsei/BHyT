# Copyright 2025 the LlamaFactory team.
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

import json
import os
import signal
import sys
import time
import wandb
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key).clone() for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"), output_dir=output_dir, safe_serialization=args.save_safetensors
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=args.save_safetensors,
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "BHyT_SFT_EXP")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

def _safe_call_update_scaler(layer):
    try:
        import deepspeed
        params = [layer.self_attn.v_proj.weight, layer.self_attn.o_proj.weight]
        with deepspeed.zero.GatheredParameters(params, modifier_rank=None), torch.no_grad():
            layer.update_scaler()
        return
    except Exception:
        pass

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(layer, writeback=False, recurse=False):
            with torch.no_grad():
                layer.update_scaler()
        return
    except Exception:
        pass

    with torch.no_grad():
        layer.update_scaler()


def _broadcast_scalar_tensor(x: torch.Tensor):
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(x, src=0)


def _unwrap_to_decoder_layers(model) -> list:
    """DeepSpeed / FSDP 등 wrapping을 재귀적으로 풀고 decoder layers를 반환."""
    unwrapped = model
    visited = set()
    while id(unwrapped) not in visited:
        visited.add(id(unwrapped))
        if hasattr(unwrapped, "layers"):
            return list(unwrapped.layers)
        for attr in ("module", "model"):
            if hasattr(unwrapped, attr):
                unwrapped = getattr(unwrapped, attr)
                break
        else:
            break
    return []


class UpdateScalerCallback(TrainerCallback):
    def __init__(
        self,
        every_n_steps: int = None,
        min_interval: int = 1,
        max_interval: int = 1000,
        target_updates: int = 50,
        recompute_immediately: bool = True,
        sync_ddp: bool = True,
        var_aux_loss_steps: int = 1,
    ):
        self.every_n_steps = every_n_steps
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.target_updates = target_updates
        self.recompute_immediately = recompute_immediately
        self.sync_ddp = sync_ddp
        self.var_aux_loss_steps = var_aux_loss_steps
        self._var_aux_off_step = -1
        self._resolved = self.every_n_steps is not None

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._resolved:
            max_steps = state.max_steps if state.max_steps and state.max_steps > 0 else 25000
            self.every_n_steps = max(
                self.min_interval,
                min(self.max_interval, max_steps // self.target_updates),
            )
            self._resolved = True
            logger.info_rank0(
                f"UpdateScalerCallback: auto interval = {self.every_n_steps} steps "
                f"(max_steps={state.max_steps}, target_updates={self.target_updates})"
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step % self.every_n_steps != 0:
            return control

        model = kwargs["model"]
        layers = _unwrap_to_decoder_layers(model)

        do_compute = True
        if torch.distributed.is_initialized():
            do_compute = torch.distributed.get_rank() == 0

        if self.recompute_immediately:
            for layer in layers:
                if hasattr(layer, "update_scaler"):
                    if do_compute:
                        _safe_call_update_scaler(layer)
            if self.sync_ddp and torch.distributed.is_initialized():
                for layer in layers:
                    c = getattr(layer, "c_attn", None)
                    if isinstance(c, torch.Tensor):
                        _broadcast_scalar_tensor(c)
        else:
            for layer in layers:
                if hasattr(layer, "_c_attn_ready"):
                    layer._c_attn_ready = False

        # Activate var_aux_loss for the next N steps after c_scaler update
        if self.var_aux_loss_steps > 0:
            has_aux = any(getattr(l, 'var_aux_loss_weight', 0) > 0
                         or getattr(getattr(l, 'config', None), 'var_aux_loss_weight', 0) > 0
                         for l in layers[:1])
            # Check via _var_aux_loss_active attribute existence
            if hasattr(layers[0], '_var_aux_loss_active'):
                for layer in layers:
                    layer._var_aux_loss_active = True
                self._var_aux_off_step = step + self.var_aux_loss_steps
                logger.info_rank0(
                    f"UpdateScalerCallback: var_aux_loss ON for steps {step+1}~{self._var_aux_off_step}"
                )

        logger.info_rank0(f"UpdateScalerCallback: step {step}, updated {len(layers)} layers")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        # Turn off var_aux_loss after the scheduled steps
        if self._var_aux_off_step > 0 and state.global_step >= self._var_aux_off_step:
            model = kwargs["model"]
            layers = _unwrap_to_decoder_layers(model)
            for layer in layers:
                if hasattr(layer, '_var_aux_loss_active'):
                    layer._var_aux_loss_active = False
                    layer._var_aux_loss = None
            self._var_aux_off_step = -1
        return control

# class LayerMonitoringCallbackForBHyT(TrainerCallback):
#     r"""
#     A callback for monitoring and logging layer-wise statistics during training.

#     This callback logs the following metrics as lists to WandB:
#     1. Average magnitude and variance of the main layer's output.
#     2. L2 norm of the gradient flowing into the main layer's input.
#     3. L2 norm of the gradient flowing into the inputs of `input_layernorm` and `post_attention_layernorm`.
#     4. Learnable scaler (self.lam) values from input_layernorm and post_attention_layernorm.

#     This callback logs at the first step and every 1000 steps thereafter.
#     """

#     def __init__(self):
#         super().__init__()
#         self.forward_stats = {}
#         self.backward_stats = {}
#         self.layernorm_grad_stats = {}
#         self.hook_handles = []

#     def _forward_hook_fn(self, layer_idx):
#         """Factory function to create a forward hook for the main layer output."""
#         def hook(module, input, output):
#             hidden_states = output[0].detach()
#             magnitudes = torch.norm(hidden_states, p=2, dim=-1)
#             variances = torch.var(hidden_states, dim=-1, unbiased=False)
#             self.forward_stats[layer_idx] = {
#                 'output_avg_magnitude': magnitudes.mean().item(),
#                 'output_avg_variance': variances.mean().item(),
#             }
#         return hook

#     def _backward_hook_fn(self, layer_idx):
#         """Factory function to create a backward hook for the main layer input."""
#         def hook(module, grad_input, grad_output):
#             if grad_input[0] is not None:
#                 self.backward_stats[layer_idx] = {
#                     'input_grad_norm': torch.norm(grad_input[0], p=2).item(),
#                 }
#         return hook

#     def _layernorm_backward_hook_fn(self, layer_idx, norm_type):
#         """Factory function to create a backward hook for LayerNorm inputs."""
#         def hook(module, grad_input, grad_output):
#             if grad_input[0] is not None:
#                 if layer_idx not in self.layernorm_grad_stats:
#                     self.layernorm_grad_stats[layer_idx] = {}
#                 grad_norm = torch.norm(grad_input[0], p=2).item()
#                 self.layernorm_grad_stats[layer_idx][norm_type] = grad_norm
#         return hook

#     @override
#     def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         """Register all necessary hooks at the beginning of training."""
#         if not state.is_world_process_zero or "wandb" not in args.report_to:
#             return

#         model = kwargs.get("model")
#         if model is None: return
        
#         try:
#             layers = model.model.layers
#         except AttributeError:
#             logger.warning("Could not find 'model.layers' in the model structure. Disabling callback.")
#             return

#         logger.info(f"Registering hooks for {len(layers)} layers for monitoring.")
#         for i, layer in enumerate(layers):
#             f_handle = layer.register_forward_hook(self._forward_hook_fn(i))
#             self.hook_handles.append(f_handle)
#             b_handle = layer.register_full_backward_hook(self._backward_hook_fn(i))
#             self.hook_handles.append(b_handle)

#             in_ln_handle = layer.input_layernorm.register_full_backward_hook(
#                 self._layernorm_backward_hook_fn(i, "input_layernorm")
#             )
#             self.hook_handles.append(in_ln_handle)
#             post_ln_handle = layer.post_attention_layernorm.register_full_backward_hook(
#                 self._layernorm_backward_hook_fn(i, "post_attention_layernorm")
#             )
#             self.hook_handles.append(post_ln_handle)

#     @override
#     def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         """Collect statistics and log them as lists to WandB every 1000 steps."""
#         if not state.is_world_process_zero or "wandb" not in args.report_to:
#             return
        
#         if state.global_step > 1 and state.global_step % 1000 != 0:
#             return
        
#         logger.info(f"Logging layer metrics at step {state.global_step}.")
        
#         model = kwargs.get("model")
#         if model is None: return
#         try:
#             layers = model.model.layers
#             num_layers = len(layers)
#         except AttributeError:
#             return

#         log_data = {}
#         metrics = {
#             "layer_idx": [],
#             "output_avg_magnitude": [], "output_avg_variance": [], "input_grad_norm": [],
#             "input_layernorm_grad_norm": [], "post_attention_layernorm_grad_norm": [],
#             "input_layernorm_lam": [], "post_attention_layernorm_lam": [],
#         }

#         for i in range(num_layers):
#             layer = layers[i]
#             metrics["layer_idx"].append(i)
#             metrics["output_avg_magnitude"].append(self.forward_stats.get(i, {}).get("output_avg_magnitude"))
#             metrics["output_avg_variance"].append(self.forward_stats.get(i, {}).get("output_avg_variance"))
#             metrics["input_grad_norm"].append(self.backward_stats.get(i, {}).get("input_grad_norm"))
            
#             ln_stats = self.layernorm_grad_stats.get(i, {})
#             metrics["input_layernorm_grad_norm"].append(ln_stats.get("input_layernorm"))
#             metrics["post_attention_layernorm_grad_norm"].append(ln_stats.get("post_attention_layernorm"))
            
#             # Log learnable scaler (lam) values from layernorms
#             input_lam = None
#             post_lam = None
#             if hasattr(layer.input_layernorm, 'lam'):
#                 input_lam = layer.input_layernorm.lam.detach().item()
#             if hasattr(layer.post_attention_layernorm, 'lam'):
#                 post_lam = layer.post_attention_layernorm.lam.detach().item()
            
#             metrics["input_layernorm_lam"].append(input_lam)
#             metrics["post_attention_layernorm_lam"].append(post_lam)

#         for metric_name, values in metrics.items():
#             if any(v is not None for v in values):
#                 log_data[f"{metric_name}"] = values
        
#         if log_data:
#             for i in range(len(log_data["layer_idx"])):
#                 wandb.log({
#                     "layer_idx": log_data["layer_idx"][i],
#                     f"step_{state.global_step}_output_avg_magnitude": log_data["output_avg_magnitude"][i],
#                     f"step_{state.global_step}_output_avg_variance": log_data["output_avg_variance"][i],
#                     f"step_{state.global_step}_input_grad_norm": log_data["input_grad_norm"][i],
#                     f"step_{state.global_step}_input_layernorm_grad_norm": log_data["input_layernorm_grad_norm"][i],
#                     f"step_{state.global_step}_post_attention_layernorm_grad_norm": log_data["post_attention_layernorm_grad_norm"][i],
#                     f"step_{state.global_step}_input_layernorm_lam": log_data["input_layernorm_lam"][i],
#                     f"step_{state.global_step}_post_attention_layernorm_lam": log_data["post_attention_layernorm_lam"][i],
#                 })
#             # wandb.log(log_data)

#         self.forward_stats.clear()
#         self.backward_stats.clear()
#         self.layernorm_grad_stats.clear()

#     @override
#     def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         """Remove all hooks to clean up."""
#         for handle in self.hook_handles:
#             handle.remove()
#         self.hook_handles = []
#         logger.info("Removed all monitoring hooks.")

# class LayerMonitoringCallback(TrainerCallback):
#     r"""
#     A callback for monitoring and logging layer-wise statistics during training.

#     This callback logs the following metrics as lists to WandB:
#     1. Average magnitude and variance of the main layer's output.
#     2. L2 norm of the gradient flowing into the main layer's input.
#     3. L2 norm of the gradient flowing into the inputs of `input_layernorm` and `post_attention_layernorm`.

#     This callback logs at the first step and every 1000 steps thereafter.
#     """

#     def __init__(self):
#         super().__init__()
#         self.forward_stats = {}
#         self.backward_stats = {}
#         self.layernorm_grad_stats = {}
#         self.hook_handles = []

#     def _forward_hook_fn(self, layer_idx):
#         """Factory function to create a forward hook for the main layer output."""
#         def hook(module, input, output):
#             hidden_states = output[0].detach()
#             magnitudes = torch.norm(hidden_states, p=2, dim=-1)
#             variances = torch.var(hidden_states, dim=-1, unbiased=False)
#             self.forward_stats[layer_idx] = {
#                 'output_avg_magnitude': magnitudes.mean().item(),
#                 'output_avg_variance': variances.mean().item(),
#             }
#         return hook

#     def _backward_hook_fn(self, layer_idx):
#         """Factory function to create a backward hook for the main layer input."""
#         def hook(module, grad_input, grad_output):
#             if grad_input[0] is not None:
#                 self.backward_stats[layer_idx] = {
#                     'input_grad_norm': torch.norm(grad_input[0], p=2).item(),
#                 }
#         return hook

#     def _layernorm_backward_hook_fn(self, layer_idx, norm_type):
#         """Factory function to create a backward hook for LayerNorm inputs."""
#         def hook(module, grad_input, grad_output):
#             if grad_input[0] is not None:
#                 if layer_idx not in self.layernorm_grad_stats:
#                     self.layernorm_grad_stats[layer_idx] = {}
#                 grad_norm = torch.norm(grad_input[0], p=2).item()
#                 self.layernorm_grad_stats[layer_idx][norm_type] = grad_norm
#         return hook

#     @override
#     def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         """Register all necessary hooks at the beginning of training."""
#         if not state.is_world_process_zero or "wandb" not in args.report_to:
#             return

#         model = kwargs.get("model")
#         if model is None: return
        
#         try:
#             layers = model.model.layers
#         except AttributeError:
#             logger.warning("Could not find 'model.layers' in the model structure. Disabling callback.")
#             return

#         logger.info(f"Registering hooks for {len(layers)} layers for monitoring.")
#         for i, layer in enumerate(layers):
#             f_handle = layer.register_forward_hook(self._forward_hook_fn(i))
#             self.hook_handles.append(f_handle)
#             b_handle = layer.register_full_backward_hook(self._backward_hook_fn(i))
#             self.hook_handles.append(b_handle)

#             in_ln_handle = layer.input_layernorm.register_full_backward_hook(
#                 self._layernorm_backward_hook_fn(i, "input_layernorm")
#             )
#             self.hook_handles.append(in_ln_handle)
#             post_ln_handle = layer.post_attention_layernorm.register_full_backward_hook(
#                 self._layernorm_backward_hook_fn(i, "post_attention_layernorm")
#             )
#             self.hook_handles.append(post_ln_handle)

#     @override
#     def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         """Collect statistics and log them as lists to WandB every 1000 steps."""
#         if not state.is_world_process_zero or "wandb" not in args.report_to:
#             return
        
#         if state.global_step > 1 and state.global_step % 1000 != 0:
#             return
        
#         logger.info(f"Logging layer metrics at step {state.global_step}.")
        
#         model = kwargs.get("model")
#         if model is None: return
#         try:
#             num_layers = len(model.model.layers)
#         except AttributeError:
#             return

#         log_data = {}
#         metrics = {
#             "layer_idx": [],
#             "output_avg_magnitude": [], "output_avg_variance": [], "input_grad_norm": [],
#             "input_layernorm_grad_norm": [], "post_attention_layernorm_grad_norm": [],
#         }

#         for i in range(num_layers):
#             metrics["layer_idx"].append(i)
#             metrics["output_avg_magnitude"].append(self.forward_stats.get(i, {}).get("output_avg_magnitude"))
#             metrics["output_avg_variance"].append(self.forward_stats.get(i, {}).get("output_avg_variance"))
#             metrics["input_grad_norm"].append(self.backward_stats.get(i, {}).get("input_grad_norm"))
            
#             ln_stats = self.layernorm_grad_stats.get(i, {})
#             metrics["input_layernorm_grad_norm"].append(ln_stats.get("input_layernorm"))
#             metrics["post_attention_layernorm_grad_norm"].append(ln_stats.get("post_attention_layernorm"))

#         for metric_name, values in metrics.items():
#             if any(v is not None for v in values):
#                 log_data[f"{metric_name}"] = values
        
#         if log_data:
#             for i in range(len(log_data["layer_idx"])):
#                 wandb.log({
#                     "layer_idx": log_data["layer_idx"][i],
#                     f"step_{state.global_step}_output_avg_magnitude": log_data["output_avg_magnitude"][i],
#                     f"step_{state.global_step}_output_avg_variance": log_data["output_avg_variance"][i],
#                     f"step_{state.global_step}_input_grad_norm": log_data["input_grad_norm"][i],
#                     f"step_{state.global_step}_input_layernorm_grad_norm": log_data["input_layernorm_grad_norm"][i],
#                     f"step_{state.global_step}_post_attention_layernorm_grad_norm": log_data["post_attention_layernorm_grad_norm"][i],
#                 })
#             # wandb.log(log_data)

#         self.forward_stats.clear()
#         self.backward_stats.clear()
#         self.layernorm_grad_stats.clear()

#     @override
#     def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         """Remove all hooks to clean up."""
#         for handle in self.hook_handles:
#             handle.remove()
#         self.hook_handles = []
#         logger.info("Removed all monitoring hooks.")

class LayerMonitoringCallbackForBHyT(TrainerCallback):
    r"""
    A callback for monitoring and logging layer-wise statistics during training.

    This callback logs the following metrics as lists to WandB:
    1. Average magnitude and variance of the main layer's output.
    2. L2 norm of the gradient flowing into the main layer's input.
    3. L2 norm of the gradient flowing into the inputs of `input_layernorm` and `post_attention_layernorm`.
    4. Learnable scaler (self.lam) values from input_layernorm and post_attention_layernorm.

    This callback logs at the first step and every 1000 steps thereafter.
    """

    def __init__(self):
        super().__init__()
        self.forward_stats = {}
        self.backward_stats = {}
        self.layernorm_grad_stats = {}
        self.hook_handles = []

        # Timing state
        # Stores last measured time (ms) per label and phase: key = f"{label}:{phase}"
        # labels:
        #   - "model" (top-level model forward)
        #   - "decoder_layer_{i}"
        #   - "layer_{i}_input_layernorm"
        #   - "layer_{i}_post_attention_layernorm"
        # phases: "train" or "eval"
        self.timing_stats = {}
        self._timing_handles = []
        self._cuda = torch.cuda.is_available()
        self._events = {}  # label -> (start_event, end_event)
        self._cpu_timer = {}  # key -> start_time (perf_counter)

    def _forward_hook_fn(self, layer_idx):
        """Factory function to create a forward hook for the main layer output."""
        def hook(module, input, output):
            hidden_states = output[0].detach()
            magnitudes = torch.norm(hidden_states, p=2, dim=-1)
            variances = torch.var(hidden_states, dim=-1, unbiased=False)
            self.forward_stats[layer_idx] = {
                'output_avg_magnitude': magnitudes.mean().item(),
                'output_avg_variance': variances.mean().item(),
            }
        return hook

    def _backward_hook_fn(self, layer_idx):
        """Factory function to create a backward hook for the main layer input."""
        def hook(module, grad_input, grad_output):
            if grad_input[0] is not None:
                self.backward_stats[layer_idx] = {
                    'input_grad_norm': torch.norm(grad_input[0], p=2).item(),
                }
        return hook

    def _layernorm_backward_hook_fn(self, layer_idx, norm_type):
        """Factory function to create a backward hook for LayerNorm inputs."""
        def hook(module, grad_input, grad_output):
            if grad_input[0] is not None:
                if layer_idx not in self.layernorm_grad_stats:
                    self.layernorm_grad_stats[layer_idx] = {}
                grad_norm = torch.norm(grad_input[0], p=2).item()
                self.layernorm_grad_stats[layer_idx][norm_type] = grad_norm
        return hook

    def _attach_timing(self, module, label: str):
        """Attach forward pre/post hooks to measure forward latency for a module."""
        if self._cuda and label not in self._events:
            self._events[label] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

        def pre_hook(mod, inputs):
            phase = "train" if mod.training else "eval"
            key = f"{label}:{phase}"
            if self._cuda:
                # Record on current stream
                self._events[label][0].record()
            else:
                self._cpu_timer[key] = time.perf_counter()

        def post_hook(mod, inputs, output):
            phase = "train" if mod.training else "eval"
            key = f"{label}:{phase}"
            if self._cuda:
                self._events[label][1].record()
                self._events[label][1].synchronize()
                ms = self._events[label][0].elapsed_time(self._events[label][1])
            else:
                start = self._cpu_timer.pop(key, None)
                ms = (time.perf_counter() - start) * 1000.0 if start is not None else 0.0
            self.timing_stats[key] = ms

        h1 = module.register_forward_pre_hook(pre_hook)
        h2 = module.register_forward_hook(post_hook)
        self.hook_handles.append(h1)
        self.hook_handles.append(h2)
        self._timing_handles.extend([h1, h2])

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Register all necessary hooks at the beginning of training."""
        if not state.is_world_process_zero or "wandb" not in args.report_to:
            return

        model = kwargs.get("model")
        if model is None: return
        
        try:
            layers = model.model.layers
        except AttributeError:
            logger.warning("Could not find 'model.layers' in the model structure. Disabling callback.")
            return

        logger.info(f"Registering hooks for {len(layers)} layers for monitoring.")
        for i, layer in enumerate(layers):
            f_handle = layer.register_forward_hook(self._forward_hook_fn(i))
            self.hook_handles.append(f_handle)
            b_handle = layer.register_full_backward_hook(self._backward_hook_fn(i))
            self.hook_handles.append(b_handle)
            
            in_ln_handle = layer.input_layernorm.register_full_backward_hook(
                self._layernorm_backward_hook_fn(i, "input_layernorm")
            )
            self.hook_handles.append(in_ln_handle)
            post_ln_handle = layer.post_attention_layernorm.register_full_backward_hook(
                self._layernorm_backward_hook_fn(i, "post_attention_layernorm")
            )
            self.hook_handles.append(post_ln_handle)

        # Timing hooks: model-level, per decoder layer, and layernorms
        self._attach_timing(model, "model")
        for i, layer in enumerate(layers):
            self._attach_timing(layer, f"decoder_layer_{i}")
            self._attach_timing(layer.input_layernorm, f"layer_{i}_input_layernorm")
            self._attach_timing(layer.post_attention_layernorm, f"layer_{i}_post_attention_layernorm")

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Collect statistics and log them as lists to WandB every 1000 steps."""
        if not state.is_world_process_zero or "wandb" not in args.report_to:
            return
        
        if state.global_step > 1 and state.global_step % 1000 != 0:
            return
        
        logger.info(f"Logging layer metrics at step {state.global_step}.")
        
        model = kwargs.get("model")
        if model is None: return
        try:
            layers = model.model.layers
            num_layers = len(layers)
        except AttributeError:
            return

        log_data = {}
        metrics = {
            "layer_idx": [],
            "output_avg_magnitude": [], "output_avg_variance": [], "input_grad_norm": [],
            "input_layernorm_grad_norm": [], "post_attention_layernorm_grad_norm": [],
            "train_decoder_ms": [], "train_input_layernorm_ms": [], "train_post_attention_layernorm_ms": [],
            "eval_decoder_ms": [], "eval_input_layernorm_ms": [], "eval_post_attention_layernorm_ms": [],
        }

        for i in range(num_layers):
            layer = layers[i]
            metrics["layer_idx"].append(i)
            metrics["output_avg_magnitude"].append(self.forward_stats.get(i, {}).get("output_avg_magnitude"))
            metrics["output_avg_variance"].append(self.forward_stats.get(i, {}).get("output_avg_variance"))
            metrics["input_grad_norm"].append(self.backward_stats.get(i, {}).get("input_grad_norm"))
            
            ln_stats = self.layernorm_grad_stats.get(i, {})
            metrics["input_layernorm_grad_norm"].append(ln_stats.get("input_layernorm"))
            metrics["post_attention_layernorm_grad_norm"].append(ln_stats.get("post_attention_layernorm"))

            # Per-layer timings (ms)
            metrics["train_decoder_ms"].append(self.timing_stats.get(f"decoder_layer_{i}:train"))
            metrics["train_input_layernorm_ms"].append(self.timing_stats.get(f"layer_{i}_input_layernorm:train"))
            metrics["train_post_attention_layernorm_ms"].append(self.timing_stats.get(f"layer_{i}_post_attention_layernorm:train"))
            metrics["eval_decoder_ms"].append(self.timing_stats.get(f"decoder_layer_{i}:eval"))
            metrics["eval_input_layernorm_ms"].append(self.timing_stats.get(f"layer_{i}_input_layernorm:eval"))
            metrics["eval_post_attention_layernorm_ms"].append(self.timing_stats.get(f"layer_{i}_post_attention_layernorm:eval"))

        for metric_name, values in metrics.items():
            if any(v is not None for v in values):
                log_data[f"{metric_name}"] = values
        
        if log_data:
            for i in range(len(log_data["layer_idx"])):
                entry = {
                    "layer_idx": log_data["layer_idx"][i],
                    f"step_{state.global_step}_output_avg_magnitude": log_data["output_avg_magnitude"][i],
                    f"step_{state.global_step}_output_avg_variance": log_data["output_avg_variance"][i],
                    f"step_{state.global_step}_input_grad_norm": log_data["input_grad_norm"][i],
                    f"step_{state.global_step}_input_layernorm_grad_norm": log_data["input_layernorm_grad_norm"][i],
                    f"step_{state.global_step}_post_attention_layernorm_grad_norm": log_data["post_attention_layernorm_grad_norm"][i],
                    f"step_{state.global_step}_train_decoder_ms": log_data["train_decoder_ms"][i],
                    f"step_{state.global_step}_train_input_layernorm_ms": log_data["train_input_layernorm_ms"][i],
                    f"step_{state.global_step}_train_post_attention_layernorm_ms": log_data["train_post_attention_layernorm_ms"][i],
                }
                if "eval_decoder_ms" in log_data:
                    entry[f"step_{state.global_step}_eval_decoder_ms"] = log_data["eval_decoder_ms"][i]
                if "eval_input_layernorm_ms" in log_data:
                    entry[f"step_{state.global_step}_eval_input_layernorm_ms"] = log_data["eval_input_layernorm_ms"][i]
                if "eval_post_attention_layernorm_ms" in log_data:
                    entry[f"step_{state.global_step}_eval_post_attention_layernorm_ms"] = log_data["eval_post_attention_layernorm_ms"][i]
                wandb.log(entry)

            # Log model-level timing (once)
            wandb.log({
                f"step_{state.global_step}_train_model_ms": self.timing_stats.get("model:train"),
                f"step_{state.global_step}_eval_model_ms": self.timing_stats.get("model:eval"),
            })

        self.forward_stats.clear()
        self.backward_stats.clear()
        self.layernorm_grad_stats.clear()
        # Keep timing_stats for potential eval logging, don't clear here.

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        """Log eval-time timings during evaluation/prediction loops."""
        if not state.is_world_process_zero or "wandb" not in args.report_to:
            return

        # Only log if we have eval measurements
        has_eval = any(k.endswith(":eval") for k in self.timing_stats.keys())
        if not has_eval:
            return

        model = kwargs.get("model")
        if model is None: return
        try:
            layers = model.model.layers
            num_layers = len(layers)
        except AttributeError:
            return

        for i in range(num_layers):
            wandb.log({
                "layer_idx": i,
                f"step_{state.global_step}_eval_decoder_ms": self.timing_stats.get(f"decoder_layer_{i}:eval"),
                f"step_{state.global_step}_eval_input_layernorm_ms": self.timing_stats.get(f"layer_{i}_input_layernorm:eval"),
                f"step_{state.global_step}_eval_post_attention_layernorm_ms": self.timing_stats.get(f"layer_{i}_post_attention_layernorm:eval"),
            })

        wandb.log({
            f"step_{state.global_step}_eval_model_ms": self.timing_stats.get("model:eval"),
        })

        # Clear eval timings after logging
        self.timing_stats = {k: v for k, v in self.timing_stats.items() if not k.endswith(":eval")}

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Remove all hooks to clean up."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        logger.info("Removed all monitoring hooks.")

# class LayerMonitoringCallback(TrainerCallback):
#     r"""
#     A callback for monitoring and logging layer-wise statistics during training.

#     This callback logs the following metrics as lists to WandB:
#     1. Average magnitude and variance of the main layer's output.
#     2. L2 norm of the gradient flowing into the main layer's input.
#     3. L2 norm of the gradient flowing into the inputs of `input_layernorm` and `post_attention_layernorm`.

#     This callback logs at the first step and every 1000 steps thereafter.
#     """

#     def __init__(self):
#         super().__init__()
#         self.forward_stats = {}
#         self.backward_stats = {}
#         self.layernorm_grad_stats = {}
#         self.hook_handles = []

#         # Timing state (same behavior as BHyT version)
#         self.timing_stats = {}
#         self._timing_handles = []
#         self._cuda = torch.cuda.is_available()
#         self._events = {}
#         self._cpu_timer = {}

#     def _forward_hook_fn(self, layer_idx):
#         """Factory function to create a forward hook for the main layer output."""
#         def hook(module, input, output):
#             hidden_states = output[0].detach()
#             magnitudes = torch.norm(hidden_states, p=2, dim=-1)
#             variances = torch.var(hidden_states, dim=-1, unbiased=False)
#             self.forward_stats[layer_idx] = {
#                 'output_avg_magnitude': magnitudes.mean().item(),
#                 'output_avg_variance': variances.mean().item(),
#             }
#         return hook

#     def _backward_hook_fn(self, layer_idx):
#         """Factory function to create a backward hook for the main layer input."""
#         def hook(module, grad_input, grad_output):
#             if grad_input[0] is not None:
#                 self.backward_stats[layer_idx] = {
#                     'input_grad_norm': torch.norm(grad_input[0], p=2).item(),
#                 }
#         return hook

#     def _layernorm_backward_hook_fn(self, layer_idx, norm_type):
#         """Factory function to create a backward hook for LayerNorm inputs."""
#         def hook(module, grad_input, grad_output):
#             if grad_input[0] is not None:
#                 if layer_idx not in self.layernorm_grad_stats:
#                     self.layernorm_grad_stats[layer_idx] = {}
#                 grad_norm = torch.norm(grad_input[0], p=2).item()
#                 self.layernorm_grad_stats[layer_idx][norm_type] = grad_norm
#         return hook

#     def _attach_timing(self, module, label: str):
#         """Attach forward pre/post hooks to measure forward latency for a module."""
#         if self._cuda and label not in self._events:
#             self._events[label] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

#         def pre_hook(mod, inputs):
#             phase = "train" if mod.training else "eval"
#             key = f"{label}:{phase}"
#             if self._cuda:
#                 self._events[label][0].record()
#             else:
#                 self._cpu_timer[key] = time.perf_counter()

#         def post_hook(mod, inputs, output):
#             phase = "train" if mod.training else "eval"
#             key = f"{label}:{phase}"
#             if self._cuda:
#                 self._events[label][1].record()
#                 self._events[label][1].synchronize()
#                 ms = self._events[label][0].elapsed_time(self._events[label][1])
#             else:
#                 start = self._cpu_timer.pop(key, None)
#                 ms = (time.perf_counter() - start) * 1000.0 if start is not None else 0.0
#             self.timing_stats[key] = ms

#         h1 = module.register_forward_pre_hook(pre_hook)
#         h2 = module.register_forward_hook(post_hook)
#         self.hook_handles.append(h1)
#         self.hook_handles.append(h2)
#         self._timing_handles.extend([h1, h2])

#     @override
#     def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         """Register all necessary hooks at the beginning of training."""
#         if not state.is_world_process_zero or "wandb" not in args.report_to:
#             return

#         model = kwargs.get("model")
#         if model is None: return
        
#         try:
#             layers = model.model.layers
#         except AttributeError:
#             logger.warning("Could not find 'model.layers' in the model structure. Disabling callback.")
#             return

#         logger.info(f"Registering hooks for {len(layers)} layers for monitoring.")
#         for i, layer in enumerate(layers):
#             f_handle = layer.register_forward_hook(self._forward_hook_fn(i))
#             self.hook_handles.append(f_handle)
#             b_handle = layer.register_full_backward_hook(self._backward_hook_fn(i))
#             self.hook_handles.append(b_handle)

#             in_ln_handle = layer.input_layernorm.register_full_backward_hook(
#                 self._layernorm_backward_hook_fn(i, "input_layernorm")
#             )
#             self.hook_handles.append(in_ln_handle)
#             post_ln_handle = layer.post_attention_layernorm.register_full_backward_hook(
#                 self._layernorm_backward_hook_fn(i, "post_attention_layernorm")
#             )
#             self.hook_handles.append(post_ln_handle)

#         # Timing hooks: model-level, per decoder layer, and layernorms
#         self._attach_timing(model, "model")
#         for i, layer in enumerate(layers):
#             self._attach_timing(layer, f"decoder_layer_{i}")
#             self._attach_timing(layer.input_layernorm, f"layer_{i}_input_layernorm")
#             self._attach_timing(layer.post_attention_layernorm, f"layer_{i}_post_attention_layernorm")

#     @override
#     def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         """Collect statistics and log them as lists to WandB every 1000 steps."""
#         if not state.is_world_process_zero or "wandb" not in args.report_to:
#             return
        
#         if state.global_step > 1 and state.global_step % 100 != 0:
#             return
        
#         logger.info(f"Logging layer metrics at step {state.global_step}.")
        
#         model = kwargs.get("model")
#         if model is None: return
#         try:
#             num_layers = len(model.model.layers)
#         except AttributeError:
#             return

#         log_data = {}
#         metrics = {
#             "layer_idx": [],
#             "output_avg_magnitude": [], "output_avg_variance": [], "input_grad_norm": [],
#             "input_layernorm_grad_norm": [], "post_attention_layernorm_grad_norm": [],
#             "train_decoder_ms": [], "train_input_layernorm_ms": [], "train_post_attention_layernorm_ms": [],
#             "eval_decoder_ms": [], "eval_input_layernorm_ms": [], "eval_post_attention_layernorm_ms": [],
#         }

#         for i in range(num_layers):
#             metrics["layer_idx"].append(i)
#             metrics["output_avg_magnitude"].append(self.forward_stats.get(i, {}).get("output_avg_magnitude"))
#             metrics["output_avg_variance"].append(self.forward_stats.get(i, {}).get("output_avg_variance"))
#             metrics["input_grad_norm"].append(self.backward_stats.get(i, {}).get("input_grad_norm"))
            
#             ln_stats = self.layernorm_grad_stats.get(i, {})
#             metrics["input_layernorm_grad_norm"].append(ln_stats.get("input_layernorm"))
#             metrics["post_attention_layernorm_grad_norm"].append(ln_stats.get("post_attention_layernorm"))

#             metrics["train_decoder_ms"].append(self.timing_stats.get(f"decoder_layer_{i}:train"))
#             metrics["train_input_layernorm_ms"].append(self.timing_stats.get(f"layer_{i}_input_layernorm:train"))
#             metrics["train_post_attention_layernorm_ms"].append(self.timing_stats.get(f"layer_{i}_post_attention_layernorm:train"))
#             metrics["eval_decoder_ms"].append(self.timing_stats.get(f"decoder_layer_{i}:eval"))
#             metrics["eval_input_layernorm_ms"].append(self.timing_stats.get(f"layer_{i}_input_layernorm:eval"))
#             metrics["eval_post_attention_layernorm_ms"].append(self.timing_stats.get(f"layer_{i}_post_attention_layernorm:eval"))

#         for metric_name, values in metrics.items():
#             if any(v is not None for v in values):
#                 log_data[f"{metric_name}"] = values
        
#         if log_data:
#             for i in range(len(log_data["layer_idx"])):
#                 entry = {
#                     "layer_idx": log_data["layer_idx"][i],
#                     f"step_{state.global_step}_output_avg_magnitude": log_data["output_avg_magnitude"][i],
#                     f"step_{state.global_step}_output_avg_variance": log_data["output_avg_variance"][i],
#                     f"step_{state.global_step}_input_grad_norm": log_data["input_grad_norm"][i],
#                     f"step_{state.global_step}_input_layernorm_grad_norm": log_data["input_layernorm_grad_norm"][i],
#                     f"step_{state.global_step}_post_attention_layernorm_grad_norm": log_data["post_attention_layernorm_grad_norm"][i],
#                     f"step_{state.global_step}_train_decoder_ms": log_data["train_decoder_ms"][i],
#                     f"step_{state.global_step}_train_input_layernorm_ms": log_data["train_input_layernorm_ms"][i],
#                     f"step_{state.global_step}_train_post_attention_layernorm_ms": log_data["train_post_attention_layernorm_ms"][i],
#                 }
#                 if "eval_decoder_ms" in log_data:
#                     entry[f"step_{state.global_step}_eval_decoder_ms"] = log_data["eval_decoder_ms"][i]
#                 if "eval_input_layernorm_ms" in log_data:
#                     entry[f"step_{state.global_step}_eval_input_layernorm_ms"] = log_data["eval_input_layernorm_ms"][i]
#                 if "eval_post_attention_layernorm_ms" in log_data:
#                     entry[f"step_{state.global_step}_eval_post_attention_layernorm_ms"] = log_data["eval_post_attention_layernorm_ms"][i]
#                 wandb.log(entry)

#             wandb.log({
#                 f"step_{state.global_step}_train_model_ms": self.timing_stats.get("model:train"),
#                 f"step_{state.global_step}_eval_model_ms": self.timing_stats.get("model:eval"),
#             })

#         self.forward_stats.clear()
#         self.backward_stats.clear()
#         self.layernorm_grad_stats.clear()
#         # Keep timing_stats for potential eval logging, don't clear here.

#     @override
#     def on_prediction_step(
#         self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
#     ):
#         """Log eval-time timings during evaluation/prediction loops."""
#         if not state.is_world_process_zero or "wandb" not in args.report_to:
#             return

#         has_eval = any(k.endswith(":eval") for k in self.timing_stats.keys())
#         if not has_eval:
#             return

#         model = kwargs.get("model")
#         if model is None: return
#         try:
#             num_layers = len(model.model.layers)
#         except AttributeError:
#             return

#         for i in range(num_layers):
#             wandb.log({
#                 "layer_idx": i,
#                 f"step_{state.global_step}_eval_decoder_ms": self.timing_stats.get(f"decoder_layer_{i}:eval"),
#                 f"step_{state.global_step}_eval_input_layernorm_ms": self.timing_stats.get(f"layer_{i}_input_layernorm:eval"),
#                 f"step_{state.global_step}_eval_post_attention_layernorm_ms": self.timing_stats.get(f"layer_{i}_post_attention_layernorm:eval"),
#             })

#         wandb.log({
#             f"step_{state.global_step}_eval_model_ms": self.timing_stats.get("model:eval"),
#         })

#         # Clear eval timings after logging
#         self.timing_stats = {k: v for k, v in self.timing_stats.items() if not k.endswith(":eval")}

#     @override
#     def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         """Remove all hooks to clean up."""
#         for handle in self.hook_handles:
#             handle.remove()
#         self.hook_handles = []
#         logger.info("Removed all monitoring hooks.")

class LayerMonitoringCallback(TrainerCallback):
    r"""
    A callback for monitoring and logging layer-wise statistics during training.

    This callback logs the following metrics as lists to WandB:
    1. Average magnitude and variance of the main layer's output.
    2. L2 norm of the gradient flowing into the main layer's input.
    3. L2 norm of the gradient flowing into the inputs of `input_layernorm` and `post_attention_layernorm`.
    4. Lambda/alpha values from BHyT/BHyT_Star/DyT normalization layers.
    5. Tanh saturation ratio (|tanh_out| > 0.95) for BHyT/BHyT_Star/DyT layers.

    Logging interval: 5 steps for SFT, 1000 steps for PT.
    """

    def __init__(self, log_interval: int = None):
        super().__init__()
        self.forward_stats = {}
        self.backward_stats = {}
        self.layernorm_grad_stats = {}
        self.hook_handles = []
        self._log_interval = log_interval

        # Forward-only timing (eval 용도 유지)
        self.timing_stats = {}
        self._timing_handles = []
        self._cuda = torch.cuda.is_available()
        self._events = {}
        self._cpu_timer = {}

        # Train(FWD+BKWD) timing 누적
        from collections import defaultdict
        self.timing_train = defaultdict(lambda: {"fwd_ms": 0.0, "bwd_ms": 0.0, "total_ms": 0.0})
        self._t_train = {}  # 라벨별 이벤트/타이머 보관용

    def _forward_hook_fn(self, layer_idx):
        """Factory function to create a forward hook for the main layer output."""
        def hook(module, input, output):
            hidden_states = output[0].detach()
            magnitudes = torch.norm(hidden_states, p=2, dim=-1)
            variances = torch.var(hidden_states, dim=-1, unbiased=False)
            self.forward_stats[layer_idx] = {
                'output_avg_magnitude': magnitudes.mean().item(),
                'output_avg_variance': variances.mean().item(),
            }
        return hook

    def _backward_hook_fn(self, layer_idx):
        """Factory function to create a backward hook for the main layer input."""
        def hook(module, grad_input, grad_output):
            if grad_input[0] is not None:
                self.backward_stats[layer_idx] = {
                    'input_grad_norm': torch.norm(grad_input[0], p=2).item(),
                }
        return hook

    def _layernorm_backward_hook_fn(self, layer_idx, norm_type):
        """Factory function to create a backward hook for LayerNorm inputs."""
        def hook(module, grad_input, grad_output):
            if grad_input[0] is not None:
                if layer_idx not in self.layernorm_grad_stats:
                    self.layernorm_grad_stats[layer_idx] = {}
                grad_norm = torch.norm(grad_input[0], p=2).item()
                self.layernorm_grad_stats[layer_idx][norm_type] = grad_norm
        return hook

    def _attach_timing(self, module, label: str):
        """Attach forward pre/post hooks to measure forward latency for a module (forward-only, eval 포함)."""
        if self._cuda and label not in self._events:
            self._events[label] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

        def pre_hook(mod, inputs):
            phase = "train" if mod.training else "eval"
            key = f"{label}:{phase}"
            if self._cuda:
                self._events[label][0].record()
            else:
                self._cpu_timer[key] = time.perf_counter()

        def post_hook(mod, inputs, output):
            phase = "train" if mod.training else "eval"
            key = f"{label}:{phase}"
            if self._cuda:
                self._events[label][1].record()
                self._events[label][1].synchronize()
                ms = self._events[label][0].elapsed_time(self._events[label][1])
            else:
                start = self._cpu_timer.pop(key, None)
                ms = (time.perf_counter() - start) * 1000.0 if start is not None else 0.0
            self.timing_stats[key] = ms

        h1 = module.register_forward_pre_hook(pre_hook)
        h2 = module.register_forward_hook(post_hook)
        self.hook_handles.append(h1)
        self.hook_handles.append(h2)
        self._timing_handles.extend([h1, h2])

    # === 추가: Train(FWD+BKWD) 시간 측정용 훅 ===
    def _attach_train_timing_with_backward(self, module, label: str):
        """
        모듈 단위의 Forward + Backward 시간(ms)을 같은 라벨로 누적 측정.
        - GPU: torch.cuda.Event
        - CPU: time.perf_counter()
        누적 저장: self.timing_train[label] = {"fwd_ms", "bwd_ms", "total_ms"}
        """
        if label not in self._t_train:
            if self._cuda:
                self._t_train[label] = {
                    "fwd_start": torch.cuda.Event(enable_timing=True),
                    "fwd_end":   torch.cuda.Event(enable_timing=True),
                    "bwd_end":   torch.cuda.Event(enable_timing=True),
                }
            else:
                self._t_train[label] = {
                    "cpu_fwd_start": None,
                    "cpu_fwd_end":   None,
                }

        def pre_fwd(mod, inputs):
            if not mod.training:
                return
            if self._cuda:
                self._t_train[label]["fwd_start"].record()
            else:
                self._t_train[label]["cpu_fwd_start"] = time.perf_counter()

        def post_fwd(mod, inputs, output):
            if not mod.training:
                return
            if self._cuda:
                self._t_train[label]["fwd_end"].record()
                # sync는 backward에서 한 번에
            else:
                self._t_train[label]["cpu_fwd_end"] = time.perf_counter()

        def full_bwd(mod, grad_input, grad_output):
            # backward가 끝날 때 호출 (여러 번 호출될 수 있어 누적)
            if not mod.training:
                return
            if self._cuda:
                self._t_train[label]["bwd_end"].record()
                self._t_train[label]["bwd_end"].synchronize()
                fwd_ms = self._t_train[label]["fwd_start"].elapsed_time(self._t_train[label]["fwd_end"])
                bwd_ms = self._t_train[label]["fwd_end"].elapsed_time(self._t_train[label]["bwd_end"])
            else:
                t0 = self._t_train[label]["cpu_fwd_start"]
                t1 = self._t_train[label]["cpu_fwd_end"]
                t2 = time.perf_counter()
                fwd_ms = (t1 - t0) * 1000.0 if (t0 is not None and t1 is not None) else 0.0
                bwd_ms = (t2 - t1) * 1000.0 if (t1 is not None) else 0.0
            self.timing_train[label]["fwd_ms"]   += fwd_ms
            self.timing_train[label]["bwd_ms"]   += bwd_ms
            self.timing_train[label]["total_ms"] += (fwd_ms + bwd_ms)

        h1 = module.register_forward_pre_hook(pre_fwd)
        h2 = module.register_forward_hook(post_fwd)
        h3 = module.register_full_backward_hook(full_bwd)
        self.hook_handles.extend([h1, h2, h3])

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Register all necessary hooks at the beginning of training."""
        if not state.is_world_process_zero or "wandb" not in args.report_to:
            return

        # Auto-determine log interval: 5 steps for SFT (<500 steps), 1000 for PT
        if self._log_interval is None:
            max_steps = state.max_steps if state.max_steps and state.max_steps > 0 else 10000
            self._log_interval = 5 if max_steps <= 500 else 1000
        logger.info(f"LayerMonitoringCallback: log_interval = {self._log_interval} steps (max_steps={state.max_steps})")

        model = kwargs.get("model")
        if model is None: return

        try:
            layers = model.model.layers
        except AttributeError:
            logger.warning("Could not find 'model.layers' in the model structure. Disabling callback.")
            return

        logger.info(f"Registering hooks for {len(layers)} layers for monitoring.")
        for i, layer in enumerate(layers):
            f_handle = layer.register_forward_hook(self._forward_hook_fn(i))
            self.hook_handles.append(f_handle)
            b_handle = layer.register_full_backward_hook(self._backward_hook_fn(i))
            self.hook_handles.append(b_handle)

            in_ln_handle = layer.input_layernorm.register_full_backward_hook(
                self._layernorm_backward_hook_fn(i, "input_layernorm")
            )
            self.hook_handles.append(in_ln_handle)
            post_ln_handle = layer.post_attention_layernorm.register_full_backward_hook(
                self._layernorm_backward_hook_fn(i, "post_attention_layernorm")
            )
            self.hook_handles.append(post_ln_handle)

        # === Forward-only timing (eval 포함) 유지 ===
        self._attach_timing(model, "model")
        for i, layer in enumerate(layers):
            self._attach_timing(layer, f"decoder_layer_{i}")
            self._attach_timing(layer.input_layernorm, f"layer_{i}_input_layernorm")
            self._attach_timing(layer.post_attention_layernorm, f"layer_{i}_post_attention_layernorm")

        # === Train(FWD+BKWD) timing: 요청 대상에 부착 ===
        # 모델 전체도 보고 싶으면 아래 줄 주석 해제
        # self._attach_train_timing_with_backward(model, "model")
        for i, layer in enumerate(layers):
            # LlamaDecoderLayer 전체
            self._attach_train_timing_with_backward(layer, f"decoder_layer_{i}")
            # LayerNorm 2종
            self._attach_train_timing_with_backward(layer.input_layernorm, f"layer_{i}_input_layernorm")
            self._attach_train_timing_with_backward(layer.post_attention_layernorm, f"layer_{i}_post_attention_layernorm")

    def _collect_tanh_stats(self, layers):
        """Collect lambda/alpha values and tanh saturation ratios from BHyT/BHyT_Star/DyT layers."""
        tanh_stats = {}
        for i, layer in enumerate(layers):
            stats = {}
            for norm_name in ["input_layernorm", "post_attention_layernorm"]:
                norm = getattr(layer, norm_name, None)
                if norm is None:
                    continue
                cls_name = type(norm).__name__
                if cls_name in ("BHyT", "BHyT_Star"):
                    stats[f"{norm_name}_lam"] = norm.lam.item()
                    tanh_out = getattr(norm, "_last_tanh_out", None)
                    if tanh_out is not None:
                        sat_ratio = (tanh_out.abs() > 0.95).float().mean().item()
                        stats[f"{norm_name}_tanh_sat_ratio"] = sat_ratio
                elif cls_name == "DyT":
                    stats[f"{norm_name}_alpha"] = norm.alpha.item()
                    tanh_out = getattr(norm, "_last_tanh_out", None)
                    if tanh_out is not None:
                        sat_ratio = (tanh_out.abs() > 0.95).float().mean().item()
                        stats[f"{norm_name}_tanh_sat_ratio"] = sat_ratio
            if stats:
                tanh_stats[i] = stats
        return tanh_stats

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Collect statistics and log them to WandB."""
        if not state.is_world_process_zero or "wandb" not in args.report_to:
            return

        log_interval = self._log_interval or 1000
        if state.global_step > 1 and state.global_step % log_interval != 0:
            return

        logger.info(f"Logging layer metrics at step {state.global_step}.")

        model = kwargs.get("model")
        if model is None: return
        try:
            layers = model.model.layers
            num_layers = len(layers)
        except AttributeError:
            return

        # === Collect tanh/lambda stats ===
        tanh_stats = self._collect_tanh_stats(layers)

        log_data = {}
        metrics = {
            "layer_idx": [],
            "output_avg_magnitude": [], "output_avg_variance": [], "input_grad_norm": [],
            "input_layernorm_grad_norm": [], "post_attention_layernorm_grad_norm": [],
            "train_decoder_ms": [], "train_input_layernorm_ms": [], "train_post_attention_layernorm_ms": [],
            "eval_decoder_ms": [], "eval_input_layernorm_ms": [], "eval_post_attention_layernorm_ms": [],
            # tanh-based norm metrics
            "input_ln_lam": [], "post_ln_lam": [],
            "input_ln_tanh_sat": [], "post_ln_tanh_sat": [],
        }

        # === 라벨 헬퍼 ===
        def t_train_total(label):
            t = self.timing_train.get(label, None)
            return (t["total_ms"] if t is not None else None)

        def t_eval_forward(label):
            return self.timing_stats.get(f"{label}:eval")

        # === 수집 ===
        for i in range(num_layers):
            metrics["layer_idx"].append(i)
            metrics["output_avg_magnitude"].append(self.forward_stats.get(i, {}).get("output_avg_magnitude"))
            metrics["output_avg_variance"].append(self.forward_stats.get(i, {}).get("output_avg_variance"))
            metrics["input_grad_norm"].append(self.backward_stats.get(i, {}).get("input_grad_norm"))

            ln_stats = self.layernorm_grad_stats.get(i, {})
            metrics["input_layernorm_grad_norm"].append(ln_stats.get("input_layernorm"))
            metrics["post_attention_layernorm_grad_norm"].append(ln_stats.get("post_attention_layernorm"))

            # === Train(FWD+BKWD total) ===
            metrics["train_decoder_ms"].append(t_train_total(f"decoder_layer_{i}"))
            metrics["train_input_layernorm_ms"].append(t_train_total(f"layer_{i}_input_layernorm"))
            metrics["train_post_attention_layernorm_ms"].append(t_train_total(f"layer_{i}_post_attention_layernorm"))

            # === Eval(forward-only) ===
            metrics["eval_decoder_ms"].append(t_eval_forward(f"decoder_layer_{i}"))
            metrics["eval_input_layernorm_ms"].append(t_eval_forward(f"layer_{i}_input_layernorm"))
            metrics["eval_post_attention_layernorm_ms"].append(t_eval_forward(f"layer_{i}_post_attention_layernorm"))

            # === tanh-based norm: lam/alpha and saturation ===
            ts = tanh_stats.get(i, {})
            # lam or alpha (unified key)
            metrics["input_ln_lam"].append(ts.get("input_layernorm_lam", ts.get("input_layernorm_alpha")))
            metrics["post_ln_lam"].append(ts.get("post_attention_layernorm_lam", ts.get("post_attention_layernorm_alpha")))
            metrics["input_ln_tanh_sat"].append(ts.get("input_layernorm_tanh_sat_ratio"))
            metrics["post_ln_tanh_sat"].append(ts.get("post_attention_layernorm_tanh_sat_ratio"))

        for metric_name, values in metrics.items():
            if any(v is not None for v in values):
                log_data[f"{metric_name}"] = values

        if log_data:
            for i in range(len(log_data["layer_idx"])):
                entry = {
                    "layer_idx": log_data["layer_idx"][i],
                    f"step_{state.global_step}_output_avg_magnitude": log_data["output_avg_magnitude"][i],
                    f"step_{state.global_step}_output_avg_variance": log_data["output_avg_variance"][i],
                    f"step_{state.global_step}_input_grad_norm": log_data["input_grad_norm"][i],
                    f"step_{state.global_step}_input_layernorm_grad_norm": log_data["input_layernorm_grad_norm"][i],
                    f"step_{state.global_step}_post_attention_layernorm_grad_norm": log_data["post_attention_layernorm_grad_norm"][i],
                    # === Train(FWD+BKWD total) 값을 기존 키에 그대로 매핑 ===
                    f"step_{state.global_step}_train_decoder_ms": log_data["train_decoder_ms"][i],
                    f"step_{state.global_step}_train_input_layernorm_ms": log_data["train_input_layernorm_ms"][i],
                    f"step_{state.global_step}_train_post_attention_layernorm_ms": log_data["train_post_attention_layernorm_ms"][i],
                }
                # === Eval(forward-only) ===
                if "eval_decoder_ms" in log_data:
                    entry[f"step_{state.global_step}_eval_decoder_ms"] = log_data["eval_decoder_ms"][i]
                if "eval_input_layernorm_ms" in log_data:
                    entry[f"step_{state.global_step}_eval_input_layernorm_ms"] = log_data["eval_input_layernorm_ms"][i]
                if "eval_post_attention_layernorm_ms" in log_data:
                    entry[f"step_{state.global_step}_eval_post_attention_layernorm_ms"] = log_data["eval_post_attention_layernorm_ms"][i]
                # === tanh-based norm metrics ===
                if "input_ln_lam" in log_data and log_data["input_ln_lam"][i] is not None:
                    entry[f"step_{state.global_step}_input_ln_lam"] = log_data["input_ln_lam"][i]
                if "post_ln_lam" in log_data and log_data["post_ln_lam"][i] is not None:
                    entry[f"step_{state.global_step}_post_ln_lam"] = log_data["post_ln_lam"][i]
                if "input_ln_tanh_sat" in log_data and log_data["input_ln_tanh_sat"][i] is not None:
                    entry[f"step_{state.global_step}_input_ln_tanh_sat"] = log_data["input_ln_tanh_sat"][i]
                if "post_ln_tanh_sat" in log_data and log_data["post_ln_tanh_sat"][i] is not None:
                    entry[f"step_{state.global_step}_post_ln_tanh_sat"] = log_data["post_ln_tanh_sat"][i]
                wandb.log(entry)

            wandb.log({
                f"step_{state.global_step}_eval_model_ms": self.timing_stats.get("model:eval"),
            })

        # 스텝 종료 후 상태 초기화
        self.forward_stats.clear()
        self.backward_stats.clear()
        self.layernorm_grad_stats.clear()
        self.timing_train.clear()
        # timing_stats는 eval용 forward 기록 때문에 유지

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        """Log eval-time timings during evaluation/prediction loops."""
        if not state.is_world_process_zero or "wandb" not in args.report_to:
            return

        has_eval = any(k.endswith(":eval") for k in self.timing_stats.keys())
        if not has_eval:
            return

        model = kwargs.get("model")
        if model is None: return
        try:
            num_layers = len(model.model.layers)
        except AttributeError:
            return

        for i in range(num_layers):
            wandb.log({
                "layer_idx": i,
                f"step_{state.global_step}_eval_decoder_ms": self.timing_stats.get(f"decoder_layer_{i}:eval"),
                f"step_{state.global_step}_eval_input_layernorm_ms": self.timing_stats.get(f"layer_{i}_input_layernorm:eval"),
                f"step_{state.global_step}_eval_post_attention_layernorm_ms": self.timing_stats.get(f"layer_{i}_post_attention_layernorm:eval"),
            })

        wandb.log({
            f"step_{state.global_step}_eval_model_ms": self.timing_stats.get("model:eval"),
        })

        # Clear eval timings after logging
        self.timing_stats = {k: v for k, v in self.timing_stats.items() if not k.endswith(":eval")}

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Remove all hooks to clean up."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        logger.info("Removed all monitoring hooks.")