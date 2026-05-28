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

import os
import sys
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    if model_args.model_name_or_path.split("_")[-1] == 'pt':
        model_name_or_path = model_args.model_name_or_path.split("_")[0]
        if model_args.model_name_or_path.split("-")[3].split("_")[0] == '250M':
            model_name_or_path = "meta-llama/Llama-3.2-1B"
    else:
        model_name_or_path = model_args.model_name_or_path

    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try another one
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    try:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:  # try another one
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments", norm_name: Optional[str] = None) -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)

def load_custom_config(config_name: str):
    """Load custom model configuration from YAML file."""
    import yaml
    
    # Load YAML configuration file
    if config_name == '250M':
        config_path = "./config/llama_250M_config.yaml"
    elif config_name == '374M':
        config_path = "./config/llama_374M_config.yaml"
    else:
        config_path = f"./config/norm_config_{config_name}.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config_dict = yaml.safe_load(file)
        return config_dict
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise

def find_norm_name(model_name_or_path: str) -> str:
    parts = model_name_or_path.split('/')
    target = next((p for p in parts if 'llama' in p.lower()), None)
    if target:
        toks = target.lower().split('_')
        # Match a known norm token first, so suffixes (e.g. "_paperopt") don't break parsing.
        known = ['bhytstar', 'bhytline', 'bhyt', 'rms']
        for t in toks:
            if t in known:
                return t
        return toks[-1].lower()
    else:
        return None

def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    if model_args.model_name_or_path.split("_")[-1] == 'pt':
        stage = 'pt'
        norm_name = model_args.model_name_or_path.split("_")[-2]
        config_norm = load_custom_config(norm_name)
        config_model = None
        setattr(model_args, 'model_name_or_path', model_args.model_name_or_path.split("_")[0])
        if model_args.model_size is not None:
            config_model = load_custom_config(model_args.model_size)
    elif 'saves' in model_args.model_name_or_path.split("/"):
        stage = 'sft'
        norm_name = find_norm_name(model_args.model_name_or_path)
        
        config_norm = load_custom_config(norm_name)
        config_model = None
        if model_args.model_size is not None:
            config_model = load_custom_config(model_args.model_size)
    else: stage = None; config_model = None

    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)

    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if stage == 'pt' or stage == 'sft':
        for key, value in config_norm.items():
            setattr(config, key, value)
        if config_model is not None:
            for key, value in config_model.items():
                setattr(config, key, value)
        setattr(config, 'seq_len', model_args.model_max_length)

        if 'bhyt' in config.norm_type:
            print("Set BHyT Lambda values as...")
            setattr(config, 'input_layer_lam', model_args.input_layer_lam)
            setattr(config, 'post_layer_lam', model_args.post_layer_lam)
            setattr(config, 'last_layer_lam', model_args.last_layer_lam)
            if model_args.var_approx_method is not None:
                setattr(config, 'var_approx_method', model_args.var_approx_method)
            setattr(config, 'use_gamma_var', model_args.use_gamma_var)
            setattr(config, 'var_approx_exact_layers', model_args.var_approx_exact_layers)
            if model_args.var_approx_exact_layer_list is not None:
                setattr(config, 'var_approx_exact_layer_list', model_args.var_approx_exact_layer_list)
            setattr(config, 'bhyt_kappa', model_args.bhyt_kappa)
            if model_args.bhyt_kappa_schedule is not None:
                setattr(config, 'bhyt_kappa_schedule', model_args.bhyt_kappa_schedule)
            print(f"Input Layer Lambda: {model_args.input_layer_lam}")
            print(f"Post Layer Lambda: {model_args.post_layer_lam}")
            print(f"Last Layer Lambda: {model_args.last_layer_lam}")
            print(f"Var Approx Method: {getattr(config, 'var_approx_method', None)}")
            print(f"Use Gamma Var: {model_args.use_gamma_var}")
            if model_args.var_approx_exact_layers > 0:
                print(f"Var Approx Exact Layers: {model_args.var_approx_exact_layers}")
            if model_args.var_approx_exact_layer_list is not None:
                print(f"Var Approx Exact Layer List: {model_args.var_approx_exact_layer_list}")
            if model_args.bhyt_kappa != 10.0:
                print(f"BHyT Kappa: {model_args.bhyt_kappa}")
            if model_args.bhyt_kappa_schedule is not None:
                print(f"BHyT Kappa Schedule: {model_args.bhyt_kappa_schedule}")
            if model_args.var_aux_loss_weight > 0:
                setattr(config, 'var_aux_loss_weight', model_args.var_aux_loss_weight)
                print(f"Var Aux Loss Weight: {model_args.var_aux_loss_weight}")
        if model_args.bhyt_layer_range is not None:
            setattr(config, 'bhyt_layer_range', [int(x) for x in model_args.bhyt_layer_range.split(',')])
            print(f"BHyT Layer Range: {config.bhyt_layer_range}")
        if model_args.fallback_norm_type is not None:
            setattr(config, 'fallback_norm_type', model_args.fallback_norm_type)
            print(f"Fallback Norm Type: {config.fallback_norm_type}")
        setattr(config, 'use_shared_scale', model_args.use_shared_scale)
        if model_args.use_shared_scale:
            print(f"Use Shared Scale: {model_args.use_shared_scale}")

        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        
        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
        from models.hf_src.llama.modeling_llama import LlamaForCausalLM
        
        if model_args.train_from_scratch:
            model = LlamaForCausalLM(config)
        else:
            model = LlamaForCausalLM.from_pretrained(**init_kwargs)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        
        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen2_5_omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) == "qwen2_5_omni":
                    model = model.thinker  # use part of Omni model

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
