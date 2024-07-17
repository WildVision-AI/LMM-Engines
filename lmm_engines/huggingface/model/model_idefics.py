import requests
import torch
from PIL import Image
from io import BytesIO

from transformers.image_utils import load_image

import json
import base64
from io import BytesIO

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