import torch
from PIL import Image
import json
import base64
import time
from io import BytesIO
from .model_adapter import BaseModelAdapter
from ...utils import decode_image, image_to_url
from transformers import MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from threading import Thread
import openai
import os

MLLAMA_MODEL_LIST = (
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-Guard-3-11B-Vision",
)

class MllamaAdapter(BaseModelAdapter):
    """The model adapter for Mllama models
    """
    def match(self, model_path: str):
        return model_path in MLLAMA_MODEL_LIST

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
        if "torch_dtype" not in from_pretrained_kwargs:
            from_pretrained_kwargs["torch_dtype"] = torch.bfloat16
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path, **from_pretrained_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
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

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image"}]
            },
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
        generation_kwargs["temperature"] = None
        generation_kwargs["top_p"] = None
        if "stop" in generation_kwargs:
            generation_kwargs.pop("stop")
        output = self.model.generate(**inputs, **generation_kwargs)
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
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
        # add your custom generation code here
        image = decode_image(params["prompt"]["image"]) # This image will be decoded into a PIL image
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image"}]
            },
        ]
        # add streamer
        streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
        generation_kwargs["temperature"] = None
        generation_kwargs["top_p"] = None
        if "stop" in generation_kwargs:
            generation_kwargs.pop("stop")
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
            "organization": "meta-llama",
            "model_size": None,
            "model_link": None,
        }
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    device = "cuda"
    model_adapter = MllamaAdapter()
    test_adapter(model_adapter, model_path, device)
    
"""
# local testing
python -m lmm_engines.huggingface.model.model_mllama
"""