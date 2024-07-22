"""Model adapter registration."""

import math
import os
import sys
from typing import Dict, List, Optional
import warnings

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache

import psutil
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from ..constants import CPU_ISA
from ..conversation import Conversation, get_conv_template
from ..utils import get_gpu_memory
from ...utils import decode_image
from transformers import TextIteratorStreamer
from threading import Thread

from icecream import ic

# Check an environment variable to check if we should be sharing Peft model
# weights.  When false we treat all Peft models as separate.
peft_share_base_weights = (
    os.environ.get("PEFT_SHARE_BASE_WEIGHTS", "false").lower() == "true"
)

# keep some of the text-only api
ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-instant-1",
    "claude-instant-1.2",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
)

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
    "gpt-4o",
)

LLAVA_API_MODEL_LIST = (
    "llava-next-72b",
)

class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, device:str, from_pretrained_kwargs: dict={}):
        # self.model = ...
        # self.processor = ...
        # return self.model
        pass

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")
    
    def generate(self, params:List[dict]):
        # extract params
        prompt = params["prompt"]["text"]
        image = decode_image(params["prompt"]["image"])
        
        # format generation kwargs
        default_generation_kwargs = {"max_new_tokens": 200, "temperature": 0.2, "top_p": 0.7}
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        default_generation_kwargs.update(generation_kwargs)
        generation_kwargs = default_generation_kwargs
        generation_kwargs["do_sample"] = generation_kwargs["temperature"] > 0.0
        
        # format inputs
        prompt = ... # format the prompt
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.device, self.torch_dtype)
        
        # generation
        outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return {"text": generated_text}
    
    def generate_stream(self, params:List[dict]):
        # extract params
        prompt = params["prompt"]["text"]
        image = decode_image(params["prompt"]["image"])
        
        # format generation kwargs
        default_generation_kwargs = {"max_new_tokens": 1024, "temperature": 0.0, "top_p": 1.0}
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        default_generation_kwargs.update(generation_kwargs)
        generation_kwargs = default_generation_kwargs
        generation_kwargs["do_sample"] = generation_kwargs["temperature"] > 0.0
        
        # add streamer
        streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        
        # format prompt and generation
        prompt = ... # format the prompt
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.device, self.torch_dtype)
        
        # generation
        thread = Thread(target=self.model.generate, kwargs={**inputs, **generation_kwargs})
        thread.start()
        
        generated_text = ""
        for text in streamer:
            generated_text += text
            yield {"text": generated_text}


# A global registry for all model adapters
model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")


def raise_warning_for_incompatible_cpu_offloading_configuration(
    device: str, load_8bit: bool, cpu_offloading: bool
):
    if cpu_offloading:
        if not load_8bit:
            warnings.warn(
                "The cpu-offloading feature can only be used while also using 8-bit-quantization.\n"
                "Use '--load-8bit' to enable 8-bit-quantization\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
        if not "linux" in sys.platform:
            warnings.warn(
                "CPU-offloading is only supported on linux-systems due to the limited compatability with the bitsandbytes-package\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
        if device != "cuda":
            warnings.warn(
                "CPU-offloading is only enabled when using CUDA-devices\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
    return cpu_offloading

import os
import warnings
import shutil
import torch
import threading


import torch

def get_model_name(model_path):
    # get model name
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        model_name = model_paths[-2] + "_" + model_paths[-1]
    else:
        model_name = model_paths[-1]
    
    return model_name

def load_qwen_pretrained_model(
    model_path: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
):
    from transformers.generation import GenerationConfig
    torch.manual_seed(1234)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

    model.to(device=device)

    return tokenizer, model, dtype

def load_blip_pretrained_model(
    model_path: str,
    device: str = "cuda",
):
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
    processor = InstructBlipProcessor.from_pretrained(model_path)

    model.to(device=device)

    return model, processor


def load_deepseekvl_pretrained_model(
    model_path: str,
):
    from .vlm_utils.deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
    from .vlm_utils.deepseek_vl.utils.io import load_pil_images
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return vl_gpt, tokenizer, vl_chat_processor

def load_bunny_pretrained_model(
    model_path: str,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True)
    return model, tokenizer

def load_yivl_pretrained_model(
    model_path: str,
):
    
    from .vlm_utils.yi_llava.mm_utils import (
        get_model_name_from_path,
        load_pretrained_model,
    )
    from .vlm_utils.yi_llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info

    def disable_torch_init():
        """
        Disable the redundant torch default initialization to accelerate model creation.
        """
        import torch

        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    disable_torch_init()
    
    # model_path = "/local/home/yujielu/project/Yi/model/Yi-VL-6B"
    model_path = os.path.expanduser(model_path)
    key_info["model_path"] = model_path
    get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path)
    return model, tokenizer, image_processor

def load_uform_pretrained_model(
    model_path: str,
    device: str = "cuda",
):
    from transformers import AutoProcessor
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model.to(device=device)

    return model, processor

def load_idefics_pretrained_model(
    model_path: str,
    device: str = "cuda",
):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    model_name = model_path
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    ).to(device)

    return model, processor


try:
    from .vlm_utils.llava.model import *
except ImportError as e:
    print(f"Failed to import: {e}")
    # raise
def load_llava_pretrained_model(
    model_path: str,
    device: str = "cuda",
    load_8bit: bool = False,
    **kwargs,
):
    from .vlm_utils.llava import LlavaLlamaForCausalLM, LlavaMptForCausalLM, LlavaMistralForCausalLM
    from .vlm_utils.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    # FIXME: parameters
    use_flash_attn = False

    device_map = "auto"
    model_name = get_model_name(model_path)
    model_base = None
    
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    # elif load_4bit:
    #     kwargs['load_in_4bit'] = True
    #     kwargs['quantization_config'] = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type='nf4'
    #     )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from .vlm_utils.llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            try:
                vision_tower.load_model(device_map=device_map)
            except:
                vision_tower.load_model(device_map="cuda")
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def load_model(
    model_path: str,
    device: str = "cuda",
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    revision: str = "main",
    debug: bool = False,
):
    """Load a model from Hugging Face."""
    import accelerate

    # get model adapter
    adapter = get_model_adapter(model_path)

    # Handle device mapping
    cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(
        device, load_8bit, cpu_offloading
    )
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
        if CPU_ISA in ["avx512_bf16", "amx"]:
            try:
                import intel_extension_for_pytorch as ipex

                kwargs = {"torch_dtype": torch.bfloat16}
            except ImportError:
                warnings.warn(
                    "Intel Extension for PyTorch is not installed, it can be installed to accelerate cpu inference"
                )
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
    elif device == "xpu":
        kwargs = {"torch_dtype": torch.bfloat16}
        # Try to load ipex, while it looks unused, it links into torch for xpu support
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            warnings.warn(
                "Intel Extension for PyTorch is not installed, but is required for xpu inference."
            )
    elif device == "npu":
        kwargs = {"torch_dtype": torch.float16}
        # Try to load ipex, while it looks unused, it links into torch for xpu support
        try:
            import torch_npu
        except ImportError:
            warnings.warn("Ascend Extension for PyTorch is not installed.")
    else:
        raise ValueError(f"Invalid device: {device}")

    if cpu_offloading:
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig

        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = (
                str(math.floor(psutil.virtual_memory().available / 2**20)) + "Mib"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit_fp32_cpu_offload=cpu_offloading
        )
        kwargs["load_in_8bit"] = load_8bit
    elif load_8bit:
        if num_gpus != 1:
            warnings.warn(
                "8-bit quantization is not supported for multi-gpu inference."
            )
        else:
            model, tokenizer = adapter.load_compress_model(
                model_path=model_path,
                device=device,
                torch_dtype=kwargs["torch_dtype"],
                revision=revision,
            )
            if debug:
                print(model)
            return model, tokenizer
    kwargs["revision"] = revision

    if dtype is not None:  # Overwrite dtype if it is provided in the arguments.
        kwargs["torch_dtype"] = dtype

    # Load model
    adapter.load_model(model_path, device, kwargs)
    model = adapter.model
    if model:
        if (
            device == "cpu"
            and kwargs["torch_dtype"] is torch.bfloat16
            and CPU_ISA is not None
        ):
            model = ipex.optimize(model, dtype=kwargs["torch_dtype"])

        if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device in (
            "mps",
            "xpu",
            "npu",
        ):
            try:
                model.to(device)
            except:
                warnings.warn(
                    f"Failed to move model to device {device}. Please check if the device is available."
                )

        if device == "xpu":
            model = torch.xpu.optimize(model, dtype=kwargs["torch_dtype"], inplace=True)

    if debug:
        print(model)

    return adapter


def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template."""
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)

# try:
#     from transformers import CLIPImageProcessor
# except ImportError as e:
#     print(f"Failed to import: {e}")
#     # raise
# def get_generate_stream_function(model: torch.nn.Module, model_path: str, image_processor: Optional[CLIPImageProcessor] = None, tokenizer: Optional[AutoTokenizer] = None):
def get_generate_stream_function(model_path: str):
    """Get the generate_stream function for inference."""
    from ..inference import generate_stream

    is_llavav15_stream = "llava-v1.5" in model_path.lower()
    is_llava_stream = "llava-v1.6" in model_path.lower()
    is_qwenvl_stream = "qwen-vl-chat" in model_path.lower()
    is_blip_stream = "blip" in model_path.lower()
    is_uform_stream = "uform" in model_path.lower()
    is_deepseekvl_stream = "deepseek-vl" in model_path.lower()
    is_bunny_stream = "bunny" in model_path.lower()
    is_yivl_stream = "yi-vl" in model_path.lower()
    is_idefics_stream = "idefics2-local" in model_path.lower()
    
    is_videollava_stream = "video-llava" in model_path.lower()
    is_llavanext_stream = "llava-next" in model_path.lower()
    is_videollama2_stream = "videollama2" in model_path.lower()

    ic(model_path)
    if is_llavav15_stream:
        from .model_llava import generate_stream_llava_v15
        return generate_stream_llava_v15
    elif is_llava_stream:
        from .model_llava import generate_stream_llava_v16
        return generate_stream_llava_v16
    elif is_qwenvl_stream:
        from .model_qwenvl import generate_stream_qwenvl
        return generate_stream_qwenvl
    elif is_blip_stream:
        from .model_instructblip import generate_stream_instructblip
        return generate_stream_instructblip
    elif is_uform_stream:
        from .model_uform import generate_stream_uform
        return generate_stream_uform
    elif is_deepseekvl_stream:
        from .model_deepseekvl import generate_stream_deepseekvl
        return generate_stream_deepseekvl
    elif is_bunny_stream:
        from .model_bunny import generate_stream_bunny
        return generate_stream_bunny
    # elif is_yivl_stream:
    #     from .model_yivl import generate_stream_yivl
    #     return generate_stream_yivl
    elif is_idefics_stream:
        from .model_idefics import generate_stream_idefics
        return generate_stream_idefics
    elif is_videollava_stream:
        from .model_videollava import generate_stream_videollava
        return generate_stream_videollava
    elif is_llavanext_stream:
        from .model_llavanext import generate_stream_llavanext
        return generate_stream_llavanext
    elif is_videollama2_stream:
        from .model_videollama2 import generate_stream_videollama2
        return generate_stream_videollama2
    else:
        return generate_stream


def add_model_args(parser):
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face Hub model revision identifier",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu", "npu"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per GPU for storing model weights. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
    )


def remove_parent_directory_name(model_path):
    """Remove parent directory name."""
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    return model_path.split("/")[-1]


class ClaudeAdapter(BaseModelAdapter):
    """The model adapter for Claude"""

    def match(self, model_path: str):
        return model_path in ANTHROPIC_MODEL_LIST

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        return None, None

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("claude")

class ChatGPTAdapter(BaseModelAdapter):
    """The model adapter for ChatGPT"""

    def match(self, model_path: str):
        return model_path in OPENAI_MODEL_LIST

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        # raise NotImplementedError()
        return None, None

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("chatgpt")
    

class GeminiAdapter(BaseModelAdapter):
    """The model adapter for Gemini"""

    def match(self, model_path: str):
        return model_path in ["gemini-pro", "gemini-pro-vision", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        return None, None

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bard")

class LLaVAv15Adapter(BaseModelAdapter):
    """The model adapter for liuhaotian/llava-v1.5-7b"""

    def match(self, model_path: str):
        return "llava-v1.5" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llava-v1.5")

class LLaVAv16Adapter(BaseModelAdapter):
    """The model adapter for liuhaotian/llava-v1.5-7b"""

    def match(self, model_path: str):
        return "llava-v1.6" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llava-v1.6")
    
class QwenVLAdapter(BaseModelAdapter):
    """The model adapter for QwenVL"""

    def match(self, model_path: str):
        return "qwen-vl-chat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("qwen-vl-chat")
    
class QwenVLAPIAdapter(BaseModelAdapter):
    """The model adapter for QwenVLPlus/Max"""

    def match(self, model_path: str):
        return "qwen-vl-plus" in model_path.lower() or "qwen-vl-max" in model_path.lower() or "qwen2-72b-instruct" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        # TODO: for qwen-vl api
        # raise NotImplementedError()
        return None, None
    
    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("qwen-vl-api")

class InstructBLIPAdapter(BaseModelAdapter):
    """The model adapter for InstructBLIP"""

    def match(self, model_path: str):
        return "instructblip" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("instructblip")
    
class CogVLMAdapter(BaseModelAdapter):
    """The model adapter for CogVLM"""

    def match(self, model_path: str):
        return "cogvlm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("cogvlm")
    
class MiniCPMAdapter(BaseModelAdapter):
    """The model adapter for MiniCPM-V"""

    def match(self, model_path: str):
        return "minicpm-v" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("minicpm-v") 


class UFormAdapter(BaseModelAdapter):
    """The model adapter for UForm"""

    def match(self, model_path: str):
        return "uform" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("uform")
    
class DeepSeekVLAdapter(BaseModelAdapter):
    """The model adapter for DeepSeekVL"""

    def match(self, model_path: str):
        return "deepseek-vl" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("deepseek-vl")
    
class BunnyAdapter(BaseModelAdapter):
    """The model adapter for Bunny"""

    def match(self, model_path: str):
        return "bunny" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bunny")

class YiVLAdapter(BaseModelAdapter):
    """The model adapter for YiVL"""

    def match(self, model_path: str):
        return "yi-vl" in model_path.lower() and not "plus" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("yi-vl")


class YiVLPLUSAdapter(BaseModelAdapter):
    """The model adapter for YiVLPLUS"""

    def match(self, model_path: str):
        return "yi-vl-plus" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        return None, None

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("yi-vl")

class LlavaAPIAdapter(BaseModelAdapter):
    """The model adapter for Llava api model."""

    def match(self, model_path: str):
        return model_path.lower() in LLAVA_API_MODEL_LIST

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        return None, None

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llava-api")
    
class RekaAdapter(BaseModelAdapter):
    """The model adapter for Reka"""

    def match(self, model_path: str):
        return "reka" in model_path.lower() or "creeping-phlox-20240403" in model_path.lower() or "Reka-Flash" in model_path.lower() or "Reka-Core" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        return None, None

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("reka")
    
class IdeficsAdapter(BaseModelAdapter):
    """The model adapter for Idefics"""

    def match(self, model_path: str):
        return "idefics" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        return None, None

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("idefics")

class MiniCPMAPIAdapter(BaseModelAdapter):
    """The model adapter for minicpm-llama3-v"""

    def match(self, model_path: str):
        return "minicpm-llama3-v" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        return None, None

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("minicpm-llama3-v")

class VideoLLaVAAdapter(BaseModelAdapter):
    """The model adapter for LanguageBind/Video-LLaVA-7B"""

    def match(self, model_path: str):
        return "video-llava" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("video-llava")

class LLaVANeXTAdapter(BaseModelAdapter):
    """The model adapter for lmms-lab/LLaVA-NeXT-Video-7B"""

    def match(self, model_path: str):
        return "llava-next" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llava-next")

class VideoLLaMA2Adapter(BaseModelAdapter):
    """The model adapter for DAMO-NLP-SG/VideoLLaMA2-7B"""

    def match(self, model_path: str):
        return "videollama2" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("videollama2")
    
# Note: the registration order matters.
# The one registered earlier has a higher matching priority.
register_model_adapter(ClaudeAdapter)
register_model_adapter(ChatGPTAdapter)
register_model_adapter(LlavaAPIAdapter)
register_model_adapter(GeminiAdapter)
register_model_adapter(LLaVAv15Adapter)
register_model_adapter(LLaVAv16Adapter)
register_model_adapter(QwenVLAdapter)
register_model_adapter(InstructBLIPAdapter)
register_model_adapter(CogVLMAdapter)
register_model_adapter(MiniCPMAdapter)
register_model_adapter(UFormAdapter)
register_model_adapter(DeepSeekVLAdapter)
register_model_adapter(BunnyAdapter)
register_model_adapter(YiVLAdapter)
register_model_adapter(YiVLPLUSAdapter)
register_model_adapter(RekaAdapter)
register_model_adapter(IdeficsAdapter)
register_model_adapter(MiniCPMAPIAdapter)
register_model_adapter(QwenVLAPIAdapter)
register_model_adapter(VideoLLaVAAdapter)
register_model_adapter(LLaVANeXTAdapter)
register_model_adapter(VideoLLaMA2Adapter)
# After all adapters, try the default base adapter.
register_model_adapter(BaseModelAdapter)


## New
from .model_tinyllava import TinyLLaVAAdapter
register_model_adapter(TinyLLaVAAdapter)