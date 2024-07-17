import torch
from PIL import Image
import json
import base64
from io import BytesIO
from icecream import ic

@torch.inference_mode()
def generate_stream_tinyllava(model, tokenizer, image_processor, params, device, context_len, stream_interval, judge_sent_end=False):
    pipe = model
    
    prompt = params["prompt"]["text"]
    im_b64 = json.loads(params["prompt"]["image"])
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file)
    
    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    do_sample = temperature > 0.0
    max_new_tokens = min(int(params.get("max_new_tokens", 200)), 200)
    
    template = "USER: <image>\n{prompt}\nASSISTANT:"
    prompt = prompt if "ASSISTANT:" in prompt else template.format(prompt=prompt)
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "do_sample": do_sample})
    index = outputs[0]["generated_text"].rfind("ASSISTANT:")
    generated_text = outputs[0]["generated_text"][index+len("ASSISTANT:")+1:]
    yield {"text": generated_text}