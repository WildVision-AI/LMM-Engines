import torch
from PIL import Image
import base64
from io import BytesIO
import json
from icecream import ic

@torch.inference_mode()
def generate_stream_uform(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = params["prompt"]["text"]
    ic(prompt)
    im_b64 = json.loads(params["prompt"]["image"])
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file)

    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    max_new_tokens = min(int(params.get("max_new_tokens", 1024)), 1024)
    
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        eos_token_id=151645,
        pad_token_id=processor.tokenizer.pad_token_id
        )
    
    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True)[0]
    generated_text = decoded_text
    yield {"text": generated_text}
