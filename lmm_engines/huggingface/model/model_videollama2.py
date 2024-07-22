import torch
from PIL import Image
import json
import base64
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer, LlavaForConditionalGeneration
from threading import Thread
from typing import List

from .vlm_utils.videollama2.conversation import conv_templates
from .vlm_utils.videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from .vlm_utils.videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from .vlm_utils.videollama2.model.builder import load_pretrained_model

class VideoLLaMA2Adapter(BaseModelAdapter):
    """The model adapter for DAMO-NLP-SG/VideoLLaMA2-7B"""

    def match(self, model_path: str):
        return "videollama2" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("videollama2")
    
    def load_model(self, model_path: str, device: str, from_pretrained_kwargs: dict = ...):
        pass
    
    def generate(self, params:List[dict]):
        pass
    
    def generate_stream(self, params:List[dict]):
        pass

@torch.inference_mode()
def generate_stream_videollama2(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
    prompt = params["prompt"]["text"]
    prompt = prompt.strip("[INST]").strip("[/INST]").strip(" ")

    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    do_sample = temperature > 0.0
    max_new_tokens = min(int(params.get("max_new_tokens", 200)), 200)
        
    import json
    encoded_images = json.loads(params["prompt"]["video"])
    
    vision_input = []
    for i, im_b64 in enumerate(encoded_images):
        im_bytes = base64.b64decode(im_b64)
        im_file = BytesIO(im_bytes)
        img = Image.open(im_file)
        vision_input.append(img)

    print(">>> generate_stream_videollama2")

    modal_list = ['video']
    default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]

    video = process_video(vision_input, processor, model.config.image_aspect_ratio).to(dtype=torch.float16, device='cuda', non_blocking=True)
    video = [video]
    
    
    conv_mode = 'llama_2'
    question = default_mm_token + "\n" + prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to('cuda:0')

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images_or_videos=video,
            modal_list=modal_list,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    yield {"text": outputs[0]}
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "..."
    device = "cuda:0"
    from_pretrained_kwargs = {"torch_dtype": torch.float16}
    model_adapter = VideoLLaMA2Adapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter)
    
"""
python -m lmm_engines.huggingface.model.model_videollama2
"""