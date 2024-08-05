#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from . import *
from ..constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from typing import Optional
from torch import nn

def resize_token_embeddings(
        model, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = model._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        if hasattr(model.config, "text_config"):
            if isinstance(model.config.text_config, dict):
                model.config.text_config["vocab_size"] = model_embeds.weight.shape[0]
            else:
                model.config.text_config.vocab_size = model_embeds.weight.shape[0]
        # TODO: to be removed after v4.42, config.vocab_size is deprecated for models that have a config.text_config
        model.config.vocab_size = model_embeds.weight.shape[0]
        model.vocab_size = model_embeds.weight.shape[0]

        # Tie weights again if needed
        model.tie_weights()

        return model_embeds


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", overwrite_config=None, attn_implementation="flash_attention_2", **kwargs):
    kwargs["device_map"] = device_map

    # import pdb;pdb.set_trace()
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        # kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        kwargs["torch_dtype"] = torch.float16

    if "llava" in model_name.lower():
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("Loading LLaVA from base model...")
            if "mixtral" in model_name.lower():
                model = LlavaMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, use_flash_attention_2=False, **kwargs)
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
            non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        elif model_base is not None:
            # this may be mm projector only
            print("Loading LLaVA from base model...")
            if "mpt" in model_name.lower().replace("prompt", ""):
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(os.path.join(model_base, "configuration_mpt.py"), os.path.join(model_path, "configuration_mpt.py"))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if overwrite_config is not None:
                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif "mixtral" in model_name.lower() and "vicuna" not in model_name.lower() and "mistral" not in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMixtralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, use_flash_attention_2=False, **kwargs)
            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                # import pdb;pdb.set_trace()
                if overwrite_config is not None:
                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=cfg_pretrained, **kwargs)
            elif "qwen" in model_name.lower() or "quyen" in model_name.lower() or "longva" in model_name.lower():
                from .language_model.llava_qwen import LlavaQwenConfig
                # import pdb;pdb.set_trace()
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if overwrite_config is not None:
                    cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                    model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=cfg_pretrained, **kwargs)
                else:
                    model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if overwrite_config is not None:
                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=cfg_pretrained, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    assert "llava" in model_name.lower(), "Only LLaVA models are supported for video chatbot."
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    # model.resize_token_embeddings(len(tokenizer))
    resize_token_embeddings(model, new_num_tokens=len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
