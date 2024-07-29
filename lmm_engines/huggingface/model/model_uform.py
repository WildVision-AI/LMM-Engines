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

class UFormAdapter(BaseModelAdapter):
    """The model adapter for UForm"""

    def match(self, model_path: str):
        return "uform" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("uform")
    
    def load_model(self, model_path: str, device: str, from_pretrained_kwargs: dict = ...):
        pass
    
    def generate(self, params:dict):
        pass
    
    def generate_stream(self, params:dict):
        pass


    
    
@torch.inference_mode()
def generate_stream_uform(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = params["prompt"]["text"]
    print(prompt)
    im_b64 = json.loads(params["prompt"]["image"])
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file)

    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    max_new_tokens = min(int(params.get("max_new_tokens", 1024)), 1024)
    
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        eos_token_id=151645,
        pad_token_id=processor.tokenizer.pad_token_id
        )
    
    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True)[0]
    generated_text = decoded_text
    yield {"text": generated_text}


if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    model_adapter = UFormAdapter
    test_adapter(model_adapter, model_path, device)
    
"""
python -m lmm_engines.huggingface.model.model_uform
"""