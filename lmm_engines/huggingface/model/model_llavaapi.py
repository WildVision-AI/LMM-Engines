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

LLAVA_API_MODEL_LIST = (
    "llava-next-72b",
)

class LlavaAPIAdapter(BaseModelAdapter):
    """The model adapter for Llava api model."""

    def match(self, model_path: str):
        return model_path.lower() in LLAVA_API_MODEL_LIST

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        return None, None

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llava-api")
    
    def generate(self, params:dict):
        pass
    
    def generate_stream(self, params:dict):
        pass
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    model_adapter = LlavaAPIAdapter
    test_adapter(model_adapter, model_path, device)
    
"""
python -m lmm_engines.huggingface.model.model_llavaapi
"""
