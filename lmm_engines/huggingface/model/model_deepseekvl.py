import torch
from PIL import Image
import json
import base64
from io import BytesIO
from icecream import ic

from transformers import AutoModelForCausalLM




@torch.inference_mode()
def generate_stream_deepseekvl(model, tokenizer, image_processor, params, device, context_len, stream_interval, judge_sent_end=False):
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
        
    # vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    # tokenizer = vl_chat_processor.tokenizer

    # vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    # vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    vl_gpt, tokenizer, vl_chat_processor = model, tokenizer, image_processor

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>{prompt}",
            "images": [""]
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    # load images and prepare for inputs
    pil_images = [image] #load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    # print(f"{prepare_inputs['sft_format'][0]}", answer)
    yield {"text": answer}