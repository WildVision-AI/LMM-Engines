import torch
from PIL import Image
import json
import base64
import time
from io import BytesIO
from .model_adapter import BaseModelAdapter
from ...utils import decode_image, image_to_url
import openai

import os
import openai

class HyperbolicAPIAdapter(BaseModelAdapter):
    """The model adapter for Hyperbolic API
    model_path is something like "hyperbolic_api/{model_name}"
    """
    def match(self, model_path: str):
        return "hyperbolic_api" in model_path.lower()
    
    def get_model_names(self):
        # remove "hyperbolic_api/" prefix
        model_names = [self.model_name.split("/")[-1]]
        return model_names
    

    def load_model(self, model_path: str, device:str="cuda", from_pretrained_kwargs: dict={}):
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
        self.client = openai.OpenAI(
            api_key=os.getenv("HYPERBOLIC_API_KEY"),
            base_url="https://api.hyperbolic.xyz/v1",
        )
        self.model = None
        assert model_path.startswith("hyperbolic_api/")
        self.model_name = model_path[len("hyperbolic_api/"):]
        return self.model
    
    def generate(self, params:dict):
        """
        generation
        Args:
            params:dict = {
                "prompt": {
                    "text": str,
                    "image": str, # base64 image
                },
                **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
            }
        Returns:
            {"text": ...}
        """
        # add your custom generation code here
        image = decode_image(params["prompt"]["image"]) # This image will be decoded into a PIL image
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        image_url = image_to_url(image)

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
            },
        ]
        openai_generation_kwargs = {
            "temperature": generation_kwargs.get("temperature", 0.7),
            "max_tokens": generation_kwargs.get("max_new_tokens", 1024),
            "top_p": generation_kwargs.get("top_p", 1.0),
        }
        print(self.model_name)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **openai_generation_kwargs,
        )
        return {"text": response.choices[0].message.content}
        
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
        # add your custom generation code here
        image = decode_image(params["prompt"]["image"]) # This image will be decoded into a PIL image
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        image_url = image_to_url(image)
        
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
            },
        ]
        openai_generation_kwargs = {
            "temperature": generation_kwargs.get("temperature", 0.7),
            "max_tokens": generation_kwargs.get("max_new_tokens", 1024),
            "top_p": generation_kwargs.get("top_p", 1.0),
        }
        print(openai_generation_kwargs)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **openai_generation_kwargs,
        )
        generated_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                generated_text += chunk.choices[0].delta.content
                # print(generated_text)
                yield {"text": generated_text}
    
    def get_info(self):
        return {
            "type": "image",
            "author": "Anonymous",
            "organization": "Hyperbolic API",
            "model_size": None,
            "model_link": None,
        }
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "hyperbolic_api/mistralai/Pixtral-12B-2409"
    device = "cpu"
    model_adapter = HyperbolicAPIAdapter()
    test_adapter(model_adapter, model_path, device)
    
    model_path = "hyperbolic_api/Qwen/Qwen2-VL-7B-Instruct"
    device = "cpu"
    model_adapter = HyperbolicAPIAdapter()
    test_adapter(model_adapter, model_path, device)
    
"""
# local testing
python -m lmm_engines.huggingface.model.model_hyperbolic_api
# connect to wildvision arena
bash start_worker_on_arena.sh hyperbolic_api/mistralai/Pixtral-12B-2409 41321
bash start_worker_on_arena.sh hyperbolic_api/Qwen/Qwen2-VL-7B-Instruct 41322
"""