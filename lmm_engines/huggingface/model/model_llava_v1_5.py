import torch
from PIL import Image
import json
import base64
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer, LlavaForConditionalGeneration
from threading import Thread
from typing import List


class LLaVAv15Adapter(BaseModelAdapter):
    """The model adapter for liuhaotian/llava-v1.5-7b"""

    def match(self, model_path: str):
        return "llava-v1.5" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llava-v1.5")
    
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
    model_adapter = LLaVAv15Adapter
    test_adapter(model_adapter, model_path, device)
"""
python -m lmm_engines.huggingface.model.model_llava_v1_5
"""