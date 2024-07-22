
import torch
from PIL import Image
import json
import base64
import os
import uuid
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer, LlavaForConditionalGeneration
from threading import Thread
from typing import List



class RekaAdapter(BaseModelAdapter):
    """The model adapter for Reka"""

    def match(self, model_path: str):
        return "reka" in model_path.lower() or "creeping-phlox-20240403" in model_path.lower() or "Reka-Flash" in model_path.lower() or "Reka-Core" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        pass

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("reka")
    
    def generate(self, params:List[dict]):
        pass
    
    def process(self, params:List[dict]):
        pass
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    from_pretrained_kwargs = {"torch_dtype": torch.float16}
    model_adapter = RekaAdapter
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter)
    
"""
python -m lmm_engines.huggingface.model.model_reka
"""
    