import os
import torch
from transformers import CLIPImageProcessor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import base64
from io import BytesIO
import json

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
