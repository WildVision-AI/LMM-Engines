import torch
from PIL import Image
import json
import base64
from io import BytesIO
from icecream import ic

from transformers import AutoModelForCausalLM

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