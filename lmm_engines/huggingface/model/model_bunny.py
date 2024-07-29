import torch
from PIL import Image
import json
import base64
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image
from threading import Thread
from typing import List

class BunnyAdapter(BaseModelAdapter):
    """The model adapter for Bunny"""

    def match(self, model_path: str):
        return "bunny" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bunny")
    
    def load_model(self, model_path: str, device: str, from_pretrained_kwargs: dict = ...):
        pass
    
    def generate(self, params:dict):
        pass
    
    def generate_stream(self, params:dict):
        pass

@torch.inference_mode()
def generate_stream_bunny(model, tokenizer, image_processor, params, device, context_len, stream_interval, judge_sent_end=False):
    torch.set_default_device('cuda') 
    
    prompt = params["prompt"]["text"]
    im_b64 = json.loads(params["prompt"]["image"])
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file)
    
    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    do_sample = temperature > 0.0
    max_new_tokens = min(int(params.get("max_new_tokens", 200)), 200)
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"

    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]

    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)
    # image, sample images can be found in images folder
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
    # generate
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        use_cache=True)[0]
    answer = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

    yield {"text": answer}
    
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    model_adapter = BunnyAdapter()
    test_adapter(model_adapter, model_path, device)
"""
python -m lmm_engines.huggingface.model.model_bunny
"""
