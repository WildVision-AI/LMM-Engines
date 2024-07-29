import torch
from PIL import Image
import json
import base64
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image
from transformers import CLIPImageProcessor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from threading import Thread
from typing import List

class InstructBLIPAdapter(BaseModelAdapter):
    """The model adapter for InstructBLIP"""

    def match(self, model_path: str):
        return "instructblip" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("instructblip")

    def load_model(self, model_path: str, device: str, from_pretrained_kwargs: dict = ...):
        return super().load_model(model_path, device, from_pretrained_kwargs)
    
    def generate(self, params:dict):
        pass
    
    def generate_stream(self, params:dict):
        pass


@torch.inference_mode()
def generate_stream_instructblip(model, tokenizer, image_processor, params, device, context_len, stream_interval, judge_sent_end=False):
    processor = image_processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = params["prompt"]["text"]
    im_b64 = json.loads(params["prompt"]["image"])
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file)

    image_rgb = image.convert('RGB')
    
    from icecream import ic
    ic(prompt)
    inputs = processor(images=image_rgb, text=prompt, return_tensors="pt").to(device)

    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    max_new_tokens = min(int(params.get("max_new_tokens", 1024)), 1024)
    
    outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=max_new_tokens,
            min_length=1,
            top_p=top_p,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=temperature,
    )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    yield {"text": generated_text}

if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    model_adapter = InstructBLIPAdapter()
    test_adapter(model_adapter, model_path, device)
"""
python -m lmm_engines.huggingface.model.model_instructblip
"""