import torch
from icecream import ic
from .vlm_utils.videollava.utils import disable_torch_init

from .vlm_utils.videollava.model.builder import load_pretrained_model
from .vlm_utils.videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from .vlm_utils.videollava.conversation import conv_templates, SeparatorStyle
from .vlm_utils.videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

@torch.inference_mode()
def generate_stream_videollava(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
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
    ic(">>> generate_stream_videollava")

    disable_torch_init()
    # video = '/private/home/yujielu/downloads/datasets/VideoChatGPT/Test_Videos/v__B7rGFDRIww.mp4'
    inp = prompt#'Why is this video funny?'
    # model_path = 'LanguageBind/Video-LLaVA-7B'
    # cache_dir = 'cache_dir'
    # device = 'cuda'
    # load_4bit, load_8bit = True, False
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    video_tensor = torch.stack([vision_input])
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    yield {"text": outputs}