
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

class VideoLLaVAAdapter(BaseModelAdapter):
    """The model adapter for LanguageBind/Video-LLaVA-7B"""

    def match(self, model_path: str):
        return "video-llava" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("video-llava")
    
    def load_model(self, model_path: str, device: str, from_pretrained_kwargs: dict = ...):
        pass
    
    def generate(self, params:dict):
        pass
    
    def generate_stream(self, params:dict):
        pass
    
@torch.inference_mode()
def generate_stream_videollava(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
    prompt = params["prompt"]["text"]
    
    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    do_sample = temperature > 0.0
    max_new_tokens = min(int(params.get("max_new_tokens", 200)), 200)
        
    import json
    vision_input = torch.tensor(json.loads(params["prompt"]["video"]))
    
    # conversation = [
    #     {
    #         "role": "User",
    #         "content": f"<image_placeholder>{prompt}",
    #         "images": [""]
    #     },
    #     {
    #         "role": "Assistant",
    #         "content": ""
    #     }
    # ]
    print(">>> generate_stream_videollava")

    disable_torch_init()
    # video = '/private/home/yujielu/downloads/datasets/VideoChatGPT/Test_Videos/v__B7rGFDRIww.mp4'
    inp = prompt#'Why is this video funny?'
    # model_path = 'LanguageBind/Video-LLaVA-7B'
    # cache_dir = 'cache_dir'
    # device = 'cuda'
    # load_4bit, load_8bit = True, False
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    video_tensor = torch.stack([vision_input])
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    yield {"text": outputs}
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    from_pretrained_kwargs = {"torch_dtype": torch.float16}
    model_adapter = VideoLLaVAAdapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter)
    
"""
python -m lmm_engines.huggingface.model.model_videollava
"""