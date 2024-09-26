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
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer, LlavaForConditionalGeneration
from threading import Thread
from typing import List
from openai import OpenAI

LLAVA_API_MODEL_LIST = (
    "llava-next-72b",
    "llava-onevision-qwen2-72b-ov-chat",
    # "llava-onevision-72b-ov",
)

class LlavaAPIAdapter(BaseModelAdapter):
    """The model adapter for Llava api model."""

    def match(self, model_path: str):
        return model_path.lower() in LLAVA_API_MODEL_LIST

    def load_model(self, model_path: str, device: str="cuda", from_pretrained_kwargs: dict={}, model_type: str="image"):
        api_key, api_base = None, None
        if "api_key" in from_pretrained_kwargs.keys():
            api_key = from_pretrained_kwargs

        self.model_name = model_path
        self.model_type = model_type

        self.model = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            base_url=api_base or "https://llava-next-endpoint.lmms-lab.com/v1"
        )

        return self.model

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llava-api")
    
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
        if params["prompt"].get("image"):
            vision_input = decode_image(params["prompt"]["image"])
        else:
            vision_input = BytesIO(base64.b64decode(json.loads(params["prompt"]["video"])))
        
        image_list = get_vision_input(vision_input)

        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")

        input_messages = [{"role": "user", "content": None}]
        input_messages[-1]["content"] = [
            {"type": "text", "text": prompt},
        ]
        if image_list:
            for image_pil in image_list:
                base64_image = convert_pil_to_base64(image_pil)
                input_messages[-1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "auto"
                    },},)


        response = self.model.chat.completions.create(
            model="llava-onevision-72b-ov", #self.model_name,
            messages=input_messages,
            max_tokens=min(int(generation_kwargs.get("max_new_tokens", 512)), 512),
            temperature=float(generation_kwargs.get("temperature", 0.3)),
            top_p = float(generation_kwargs.get("top_p", 0.7)),
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
        
        if params["prompt"].get("image"):
            vision_input = decode_image(params["prompt"]["image"])
        else:
            vision_input = BytesIO(base64.b64decode(json.loads(params["prompt"]["video"])))
        
        image_list = get_vision_input(vision_input)

        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")

        input_messages = [{"role": "user", "content": None}]
        input_messages[-1]["content"] = [
            {"type": "text", "text": prompt},
        ]
        if image_list:
            for image_pil in image_list:
                base64_image = convert_pil_to_base64(image_pil)
                input_messages[-1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "auto"
                    },},)

        
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=input_messages,
            max_tokens=min(int(generation_kwargs.get("max_new_tokens", 512)), 512),
            temperature=float(generation_kwargs.get("temperature", 0.3)),
            top_p = float(generation_kwargs.get("top_p", 0.7)),
            stream = True
        )

        generated_text = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                generated_text += content
                yield {"text": generated_text}

    def get_info(self):
        return {
            "type": "image;video",
            "author": "Anonymous",
            "organization": "Anonymous",
            "model_size": None,
            "model_link": None,
        }

    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    device = "cuda"

    model_path = "llava-onevision-72b-ov"
    model_adapter = LlavaAPIAdapter()
    test_adapter(model_adapter, model_path, device)
    
"""
bash start_worker_on_arena.sh llava-onevision-72b-ov 42001 1

bash start_worker_on_arena.sh llava-onevision-qwen2-72b-ov-chat 42001 1
"""
