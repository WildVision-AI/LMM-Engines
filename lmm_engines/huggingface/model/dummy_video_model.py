

import torch
from PIL import Image
import json
import base64
import time
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_and_save_video
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer, LlavaForConditionalGeneration
from threading import Thread
from typing import List

class DummyVideoAdapter(BaseModelAdapter):
    """The model adapter for DummyVideoModel"""

    def match(self, model_path: str):
        return "dummy_video_model" in model_path.lower()

    def load_model(self, model_path: str, device:str="cuda", from_pretrained_kwargs: dict={}):
        """
        load all the elements of the models here that will be used for your model's geneation, such as the model, tokenizer, processor, etc.
        Args:
            model_path (str): the path to the model, huggingface model id or local path
            device (str): the device to run the model on
            from_pretrained_kwargs (dict): other kwargs to pass to the from_pretrained method.
                It's better to ignore this one, and set your custom kwargs in the load_model method.
        Returns:
            model: A nn.Module model or huggingface PreTrainedModel model
        """
        self.model = None
        # model_id = model_path
        # if "torch_dtype" not in from_pretrained_kwargs:
        #     from_pretrained_kwargs["torch_dtype"] = torch.float16
        # self.torch_dtype = from_pretrained_kwargs["torch_dtype"]
        # from_pretrained_kwargs["device_map"] = device
        
        # self.processor = AutoProcessor.from_pretrained(model_id)
        # self.model = LlavaForConditionalGeneration.from_pretrained(
        #     model_path, **from_pretrained_kwargs
        # )
        return self.model
    
    def generate(self, params:dict):
        """
        generation
        Args:
            params:dict = {
                "prompt": {
                    "text": str,
                    "video": str, # base64 encoded video
                },
                **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
            }
        Returns:
            {"text": ...}
        """
        # add your custom generation code here
        # video_path = decode_and_save_video(params["prompt"]["video"]) # This will save the video to a file and return the path
        # prompt = params["prompt"]["text"]
        # generation_kwargs = params.copy()
        # generation_kwargs.pop("prompt")
        ...
        return {"text": "Hi, there! It's a dummy model here for testing video models. I'm not generating anything useful."}
        
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
        dummy_response = "Hi, there! It's a dummy model here for testing video models. I'm not generating anything useful."
        generated_text = ""
        for word in dummy_response.split():
            generated_text += word + " "
            time.sleep(0.1)
            yield {"text": generated_text}
            
    def get_status(self):
        status = super().get_status()
        status["type"] = "video"
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    from_pretrained_kwargs = {"torch_dtype": torch.float16}
    model_adapter = DummyVideoAdapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter)
    
"""
# local testing
python -m lmm_engines.huggingface.model.dummy_video_model
# connect to wildvision arena
bash start_worker_on_arena.sh dummy_video_model 41411
"""