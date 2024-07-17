import argparse
import torch

from .videollm_utils.llava.utils import disable_torch_init
from .videollm_utils.llava.conversation import conv_templates, SeparatorStyle
from .videollm_utils.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from .videollm_utils.lita.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .videollm_utils.lita.model.builder import load_pretrained_model
from .videollm_utils.lita.utils import load_image, load_video, load_video_frames

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from icecream import ic
import os
import glob
import numpy as np
import decord
from decord import VideoReader

def load_visual(visual_path, processor, visual_data_type, config):
    if visual_data_type == 'image':
        image = load_image(visual_path, processor)
        return torch.stack([image] * config.num_frames, dim=0)
    elif visual_data_type == 'video_frames':
        return load_video_frames(visual_path, processor, config.num_frames)
    elif visual_data_type == 'video':
        return load_video(visual_path, processor, config.num_frames)


@torch.inference_mode()
def generate_stream_lita(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
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
    ic(">>> generate_stream_lita")
    # Model
    disable_torch_init()

    model_path = "/private/home/yujielu/downloads/weights/lita-vicuna-v1-3-13b-finetune"
    model_name = get_model_name_from_path(model_path)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"


    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    visual_path = "/private/home/yujielu/project/WildVision-Arena/examples/dancing.mp4"
    visual_data_type = "video"
    image_tensor = load_visual(visual_path, processor, visual_data_type, model.config).unsqueeze(0).half().cuda()
    image = "<image>"

    inp = prompt

    print(f"{roles[1]}: ", end="")

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    # FIXME: a hack here due to gradio angle brackets issue: https://github.com/gradio-app/gradio/issues/7198
    outputs = outputs.replace("<", "[").replace(">", "]")
    conv.messages[-1][-1] = outputs

    yield {"text": outputs}