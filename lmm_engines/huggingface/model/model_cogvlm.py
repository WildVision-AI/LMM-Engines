
import torch
from PIL import Image
import json
import base64
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image
from transformers import CLIPImageProcessor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from threading import Thread
from typing import List

class CogVLMAdapter(BaseModelAdapter):
    """The model adapter for CogVLM"""

    def match(self, model_path: str):
        return "cogvlm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("cogvlm")
    
    def load_model(self, model_path: str, device: str, from_pretrained_kwargs: dict = ...):
        pass
    
    def generate(self, params:dict):
        pass
        
    def generate_stream(self, params:dict):
        pass
    
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    from_pretrained_kwargs = {"torch_dtype": torch.float16}
    model_adapter = CogVLMAdapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter)
"""
python -m lmm_engines.huggingface.model.model_cogvlm
"""