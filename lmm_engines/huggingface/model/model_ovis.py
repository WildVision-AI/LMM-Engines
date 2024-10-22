import torch
from PIL import Image
import json
import base64
import time
import os
from io import BytesIO
from .model_adapter import BaseModelAdapter
from ...utils import decode_image, image_to_url
from transformers import AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

OVIS_MODEL_LIST = (
    "AIDC-AI/Ovis1.6-Gemma2-9B",
    "AIDC-AI/Ovis1.6-Llama3.2-3B",
)

class OvisAdapter(BaseModelAdapter):
    """The model adapter for Ovis models
    """
    def match(self, model_path: str):
        return model_path in OVIS_MODEL_LIST

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
        from_pretrained_kwargs["torch_dtype"] = torch.bfloat16
        from_pretrained_kwargs["multimodal_max_length"] = 8192
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **from_pretrained_kwargs
        )
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
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
        text = params["prompt"]["text"]
        query = f'<image>\n{text}'
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            generated_text = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)

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
        text = params["prompt"]["text"]
        query = f'<image>\n{text}'
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]

        # add streamer
        streamer = TextIteratorStreamer(self.text_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        
        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
                streamer=streamer,
            )
            thread = Thread(target=self.model.generate, kwargs=dict(inputs=input_ids, **gen_kwargs))
            thread.start()
        
            generated_text = ""
            for text in streamer:
                generated_text += text
                yield {"text": generated_text}
            thread.join()
    
    def get_info(self):
        return {
            "type": "image",
            "author": "Anonymous",
            "organization": "Alibaba",
            "model_size": None,
            "model_link": None,
        }
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    # model_path = "AIDC-AI/Ovis1.6-Gemma2-9B"
    model_path = "AIDC-AI/Ovis1.6-Llama3.2-3B"
    device = "cuda"
    model_adapter = OvisAdapter()
    test_adapter(model_adapter, model_path, device)
    
"""
# local testing
python -m lmm_engines.huggingface.model.model_ovis
"""