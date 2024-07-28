import torch
from PIL import Image
import json
import base64
import time
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer, LlavaForConditionalGeneration
from threading import Thread
from typing import List

class DummyImageAdapter(BaseModelAdapter):
    """The model adapter for DummyImageModel"""
    

    def match(self, model_path: str):
        return "dummy_image_model" in model_path.lower()

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
        self.model = None
        # model_id = model_path
        # if "torch_dtype" not in from_pretrained_kwargs:
        #     from_pretrained_kwargs["torch_dtype"] = torch.float16
        # self.torch_dtype = from_pretrained_kwargs["torch_dtype"]
        
        # self.processor = AutoProcessor.from_pretrained(model_id)
        # self.model = LlavaForConditionalGeneration.from_pretrained(
        #     model_path, **from_pretrained_kwargs
        # )
        print(self.model)
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
        # image = decode_image(params["prompt"]["image"]) # This image will be decoded into a PIL image
        # prompt = params["prompt"]["text"]
        # generation_kwargs = params.copy()
        # generation_kwargs.pop("prompt")
        ...
        return {"text": "Hi, there! It's a dummy model here for testing image models. I'm not generating anything useful."}
        
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
        ...
        dummy_response = "Hi, there! It's a dummy model here for testing image models. I'm not generating anything useful."
        generated_text = ""
        for word in dummy_response.split():
            generated_text += word + " "
            time.sleep(0.1)
            yield {"text": generated_text}
    
    def get_info(self):
        return {
            "type": "image",
            "author": "Anonymous",
            "organization": "Anonymous",
            "model_size": None,
            "model_link": None,
        }
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    from_pretrained_kwargs = {"torch_dtype": torch.float16}
    model_adapter = DummyImageAdapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter)
    
"""
# local testing
python -m lmm_engines.huggingface.model.dummy_image_model
# connect to wildvision arena
bash start_worker_on_arena.sh dummy_image_model 41411
"""