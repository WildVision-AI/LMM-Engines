"""Model adapter registration."""

import math
import os
import sys
import os
import warnings
import shutil
import torch
import threading


import torch
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

class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, device:str, from_pretrained_kwargs: dict={}):
        """
        load all the elements of the models here that will be used for your model's geneation, such as the model, tokenizer, processor, etc.
        Args:
            model_path (str): the path to the model, huggingface model id or local path
            device (str): the device to run the model on. e.g. "cuda" or "cpu", it cannot be used to load a model, use device_map in from_pretrained_kwargs instead.
            from_pretrained_kwargs (dict): other kwargs to pass to the from_pretrained method.
                including device_map, torch_dtype, etc.
                we use device_map so that we can run the model on multiple devices
        Returns:
            model: A nn.Module model or huggingface PreTrainedModel model
        """
        print("Warning: You are using the load_model method of the BaseModelAdapter, which is not implemented. Please implement the load_model method in your model adapter.")
        self.model = None
        # self.processor = ...
        return self.model

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")
    
    def generate(self, params:dict):
        """
        params:dict = {
            "prompt": {
                "text": str,
                "image": str, # base64 image
            },
            **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
        }
        """
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
    
    def generate_stream(self, params:dict):
        """
        params:dict = {
            "prompt": {
                "text": str,
                "image": str, # base64 image
            },
            **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
        }
        """
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
            
    def get_info(self):
        return {
            "type": "image",
            "author": "Anonymous",
            "organization": "Anonymous",
            "model_size": None,
            "model_link": None,
        }


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

def load_adapter(
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


from .model_instructblip import InstructBLIPAdapter
from .model_claude import ClaudeAdapter
from .model_openai import OpenAIAdapter
from .model_cogvlm import CogVLMAdapter
from .model_llavaapi import LlavaAPIAdapter
from .model_gemini import GeminiAdapter
from .model_llava_v1_5 import LLaVAv15Adapter
from .model_llava_v1_6 import LLaVAv16Adapter
from .model_qwenvl import QwenVLAdapter
from .model_minicpm import MiniCPMAdapter
from .model_uform import UFormAdapter
from .model_deepseekvl import DeepSeekVLAdapter
from .model_bunny import BunnyAdapter
from .model_yivl import YiVLAdapter
from .model_yivlplus import YiVLPLUSAdapter
from .model_reka import RekaAdapter
from .model_idefics import IdeficsAdapter
from .model_minicpmapi import MiniCPMAPIAdapter
from .model_qwenvlapi import QwenVLAPIAdapter
from .model_videollava import VideoLLaVAAdapter
from .model_llavanext import LLaVANeXTAdapter
# from .model_videollama2 import VideoLLaMA2Adapter
from .model_tinyllava import TinyLLaVAAdapter
from .dummy_image_model import DummyImageAdapter
from .dummy_video_model import DummyVideoAdapter
from .model_llavanextvideo import LLaVANextVideoAdapter

register_model_adapter(ClaudeAdapter)
register_model_adapter(OpenAIAdapter)
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
# register_model_adapter(VideoLLaMA2Adapter)
register_model_adapter(BaseModelAdapter)
register_model_adapter(TinyLLaVAAdapter)
register_model_adapter(DummyImageAdapter)
register_model_adapter(DummyVideoAdapter)
register_model_adapter(LLaVANextVideoAdapter)