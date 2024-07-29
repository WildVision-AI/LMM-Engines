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

class IdeficsAdapter(BaseModelAdapter):
    """The model adapter for Idefics"""

    def match(self, model_path: str):
        return "idefics" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        return None, None

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("idefics")
    
    def generate(self, params:dict):
        pass
    
    def generate_stream(self, params:dict):
        pass
    
@torch.inference_mode()
def generate_stream_idefics(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
    # TODO: support multiple images
    # ======= input
    input_messages = params["prompt"]["text"]

    im_b64 = json.loads(params["prompt"]["image"])
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file)
    
    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    do_sample = temperature > 0.0
    max_new_tokens = min(int(params.get("max_new_tokens", 500)), 500)
    num_beams = 1
    # =======

    text_message = input_messages[-1]["content"]
    input_messages[-1]["content"] = [
        {"type": "image"},
        {"type": "text", "text": text_message},
    ]

    prompt = processor.apply_chat_template(input_messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}


    input_token_len = inputs["input_ids"].shape[1]
    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, num_beams=num_beams, do_sample=do_sample)
    generated_texts = processor.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)

    yield {"text": generated_texts[0]}
    
    
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    model_adapter = IdeficsAdapter()
    test_adapter(model_adapter, model_path, device)
"""
python -m lmm_engines.huggingface.model.model_idefics
"""
