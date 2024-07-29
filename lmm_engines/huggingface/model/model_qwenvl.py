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

class QwenVLAdapter(BaseModelAdapter):
    """The model adapter for QwenVL"""

    def match(self, model_path: str):
        return "qwen-vl-chat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("qwen-vl-chat")
    
    def load_model(self, model_path: str, device: str, from_pretrained_kwargs: dict = ...):
        return super().load_model(model_path, device, from_pretrained_kwargs)
    
    def generate(self, params:dict):
        pass
    
    def generate_stream(self, params:dict):
        pass
    

# def generate(self, image, question):
@torch.inference_mode()
def generate_stream_qwenvl(model, tokenizer, image_processor, params, device, context_len, stream_interval, judge_sent_end=False):
    os.makedirs("tmp_log_images", exist_ok=True)
    unique_filename = str(uuid.uuid4()) + '.png'
    tmp_img_save_path = os.path.join('tmp_log_images', unique_filename)

    prompt = params["prompt"]["text"]
    im_b64 = json.loads(params["prompt"]["image"])
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file).convert('RGB')

    image_rgb = image.convert('RGB')
    image_rgb.save(tmp_img_save_path)

    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    max_new_tokens = min(int(params.get("max_new_tokens", 1024)), 1024)
    do_sample = temperature > 0.0
    temperature = None if not do_sample else temperature
    # response, history = self.model.chat(self.tokenizer, query=query, history=None)
    # thread = Thread(target=model.chat_stream, kwargs=dict(
    #     tokenizer=tokenizer,
    #     query=query,
    #     streamer=streamer,
    #     history=None
    # ))
    # thread.start()
                    
    from icecream import ic
    ic(prompt)
    query = tokenizer.from_list_format([
        {'image': tmp_img_save_path}, # Either a local path or an url
        {'text': prompt},
    ])
    generated_text = ""
    for response in model.chat_stream(tokenizer, query, history=None, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, do_sample=do_sample):
        generated_text = response
        yield {"text": generated_text}
    
    # generated_text = ""
    # for new_text in streamer:
    #     generated_text += new_text   
    #     yield {"text": generated_text}
        
    os.remove(tmp_img_save_path)
    
    # # clean
    # gc.collect()
    # torch.cuda.empty_cache()
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    model_adapter = QwenVLAdapter()
    test_adapter(model_adapter, model_path, device)
"""
python -m lmm_engines.huggingface.model.model_qwenvl
"""