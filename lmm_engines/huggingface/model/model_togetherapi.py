import torch
from PIL import Image
import json
import base64
import os
import uuid
from io import BytesIO
import numpy as np
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image, get_vision_input, convert_pil_to_base64
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer
from threading import Thread
from typing import List
from together import Together
from openai import OpenAI

TOGETHER_API_MODEL_LIST = (
    "meta-llama/Llama-Vision-Free",
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
)

class TogetherAPIAdapter(BaseModelAdapter):
    """The model adapter for Together api model."""

    def match(self, model_path: str):
        return model_path in TOGETHER_API_MODEL_LIST

    def load_model(self, model_path: str, device: str="cuda", from_pretrained_kwargs: dict={}, model_type: str="image"):
        api_key, api_base = None, None
        if "api_key" in from_pretrained_kwargs.keys():
            api_key = from_pretrained_kwargs

        self.model_name = model_path
        self.model_type = model_type

        self.model = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=api_key or os.environ["TOGETHER_API_KEY"],
            base_url=api_base or "https://api.together.xyz/v1"
        )

        return self.model

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("together-api")
    
    def generate(self, params:dict):
        """
        generation
        Args:
            params:dict = {
                "prompt": {
                    "text": str,
                    "image": str, # base64 image
                    "video": str, # base64 encoded video
                },
                **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
            }
        Returns:
            {"text": ...}
        """
        image = decode_image(params["prompt"]["image"])
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        base64_image = convert_pil_to_base64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}, 
                    {"type": "image_url",
                    "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "auto"
                        }
                    }
                ]
            },
        ]

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        return {"text": response.choices[0].message.content}
    
    def generate_stream(self, params:dict):
        """
        generation
        Args:
            params:dict = {
                "prompt": {
                    "text": str,
                    "image": str, # base64 image
                    "video": str, # base64 encoded video
                },
                **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
            }
        Returns:
            {"text": ...}
        """
        
        image = decode_image(params["prompt"]["image"])
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        base64_image = convert_pil_to_base64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}, 
                    {"type": "image_url",
                    "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "auto"
                        }
                    }
                ]
            },
        ]

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )

        generated_text = ""
        for chunk in response:
            try:
                content = chunk.choices[0].delta.content
                if content is not None:
                    generated_text += content
                    yield {"text": generated_text}
            except:
                print("")

    def get_info(self):
        return {
            "type": "image",
            "author": "Anonymous",
            "organization": "TogetherAI",
            "model_size": None,
            "model_link": None,
        }

    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    device = "cuda"

    model_path = "meta-llama/Llama-Vision-Free"
    model_adapter = TogetherAPIAdapter()
    test_adapter(model_adapter, model_path, device)
    
"""
python -m lmm_engines.huggingface.model.model_togetherapi
"""
