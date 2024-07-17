import torch
from icecream import ic

from io import BytesIO
import base64
from PIL import Image

from .vlm_utils.llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .vlm_utils.llavavid.conversation import conv_templates, SeparatorStyle
from .vlm_utils.llavavid.model.builder import load_pretrained_model
from .vlm_utils.llavavid.utils import disable_torch_init
from .vlm_utils.llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria



@torch.inference_mode()
def generate_stream_llavanext(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
    prompt = params["prompt"]["text"]

    cur_prompt = prompt
    
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

    ic(">>> generate_stream_llavanext")

    disable_torch_init()

    video = processor.preprocess(vision_input, return_tensors="pt")["pixel_values"]
    video = video.to(model.device, dtype=torch.float16)
    video = [video]
    
    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = "vicuna_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        model.update_prompt([[cur_prompt]])
        # import pdb;pdb.set_trace()
        output_ids = model.generate(
            inputs=input_ids.to(model.device), 
            images=video, 
            attention_mask=attention_masks.to(model.device), 
            modalities="video", 
            do_sample=True, 
            temperature=temperature, 
            max_new_tokens=max_new_tokens, 
            use_cache=True, 
            stopping_criteria=[stopping_criteria]
        )
        # import pdb;pdb.set_trace()
        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, use_cache=True, stopping_criteria=[stopping_criteria])

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(f"Question: {prompt}\n")
    # print(f"Response: {outputs}\n")
    # import pdb;pdb.set_trace()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    yield {"text": outputs}