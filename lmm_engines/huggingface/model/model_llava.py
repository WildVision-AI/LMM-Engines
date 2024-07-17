import gc
import os
import json
import warnings
import shutil
import torch
from .vlm_utils.llava.conversation import conv_templates, SeparatorStyle
from .vlm_utils.llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from .vlm_utils.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .vlm_utils.llava.model import *


import torch

from transformers import TextIteratorStreamer, GenerationConfig
from threading import Thread

import base64
from io import BytesIO
from PIL import Image


@torch.inference_mode()
def generate_stream_llava_v16(model, tokenizer, image_processor, params, device, context_len, stream_interval, judge_sent_end=False):
    is_multimodal = True #'llava' in model_name.lower()
    tokenizer, model, image_processor = tokenizer, model, image_processor
    prompt = params["prompt"]["text"]
    ori_prompt = prompt
    im_b64 = json.loads(params["prompt"]["image"])
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file).convert('RGB')
    images = [image]
    num_image_tokens = 0
    
    conv_mode = "llava_next"
    if conv_mode is not None and conv_mode != conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, conv_mode, conv_mode))
    else:
        conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    inp = ori_prompt
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
    
    if images is not None and len(images) > 0 and is_multimodal:
        if len(images) > 0:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError("Number of images does not match number of <image> tokens in prompt")

            # images = [load_image_from_base64(image) for image in images]
            image_sizes = [image.size for image in images]
            images = process_images(images, image_processor, model.config)

            if type(images) is list:
                images = [image.to(model.device, dtype=torch.float16) for image in images]
            else:
                images = images.to(model.device, dtype=torch.float16)

            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
        else:
            images = None
            image_sizes = None
        image_args = {"images": images, "image_sizes": image_sizes}
    else:
        images = None
        image_args = {}

    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
    max_new_tokens = min(int(params.get("max_new_tokens", 1024)), 1024)
    # stop_str = "</s>"#params.get("stop", None)
    stop_str = None
    do_sample = True if temperature > 0.001 else False


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

    max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

    if max_new_tokens < 1:
        yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
        return

    thread = Thread(target=model.generate, kwargs=dict(
        inputs=input_ids,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True,
        **image_args
    ))
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        # if generated_text.endswith(stop_str):
        #     generated_text = generated_text[:-len(stop_str)]
        # yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"    
        yield {"text": generated_text}

    # # clean
    # gc.collect()
    # torch.cuda.empty_cache()


def generate_stream_llava_v15(model, tokenizer, image_processor, params, device, context_len, stream_interval, judge_sent_end=False):
    is_multimodal = True #'llava' in model_name.lower()
    tokenizer, model, image_processor = tokenizer, model, image_processor
    prompt = params["prompt"]["text"]
    qs = ori_prompt = prompt
    im_b64 = json.loads(params["prompt"]["image"])
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file).convert('RGB')
    
    if image_processor is not None:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv_mode = "vicuna_v1"
    if image_processor is not None:
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
    # params setting from UI
    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    max_new_tokens = min(int(params.get("max_new_tokens", 1024)), 1024)


    image_tensor = process_images([image], image_processor, model.config)
    stop_str = conv_templates[conv_mode].sep if conv_templates[conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[conv_mode].sep2
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device, non_blocking=True)
    num_beams = 1
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)
    image_args = {}
    thread = Thread(target=model.generate, kwargs=dict(
        inputs=input_ids,
        images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True,
        **image_args
    ))
    thread.start()

    # generated_text = ori_prompt
    # for new_text in streamer:
    #     generated_text += new_text
    #     if generated_text.endswith(stop_str):
    #         generated_text = generated_text[:-len(stop_str)]
    #     ic(generated_text)
    #     yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"    
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        # if generated_text.endswith(stop_str):
        #     generated_text = generated_text[:-len(stop_str)]
        # yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"    
        yield {"text": generated_text}
        
    # # clean
    # gc.collect()
    # torch.cuda.empty_cache()