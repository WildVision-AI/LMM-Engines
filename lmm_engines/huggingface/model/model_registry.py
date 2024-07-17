"""Additional information of the models."""
from collections import namedtuple, OrderedDict
from typing import List


ModelInfo = namedtuple("ModelInfo", ["simple_name", "link", "description"])


model_info = OrderedDict()


def register_model_info(
    full_names: List[str], simple_name: str, link: str, description: str
):
    info = ModelInfo(simple_name, link, description)

    for full_name in full_names:
        model_info[full_name] = info


def get_model_info(name: str) -> ModelInfo:
    if name in model_info:
        return model_info[name]
    else:
        # To fix this, please use `register_model_info` to register your model
        return ModelInfo(
            name, "", "Register the description at arena.model/model_registry.py"
        )

register_model_info(
    ["gemini-pro"],
    "Gemini",
    "https://blog.google/technology/ai/google-gemini-pro-imagen-duet-ai-update/",
    "Gemini by Google",
)

register_model_info(
    ["gemini-pro-vision"],
    "Gemini",
    "https://blog.google/technology/ai/google-gemini-pro-imagen-duet-ai-update/",
    "Gemini by Google",
)

register_model_info(
    ["gemini-1.5-flash-latest"],
    "Gemini 1.5",
    "https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/",
    "Gemini 1.5 by Google",
)

register_model_info(
    ["gemini-1.5-pro-latest"],
    "Gemini 1.5",
    "https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/",
    "Gemini 1.5 by Google",
)

register_model_info(
    ["llava-v1.5-13b"],
    "LLaVA",
    "https://huggingface.co/liuhaotian/llava-v1.5-13b",
    "LLaVA",
)

register_model_info(
    ["llava-v1.6-34b"],
    "LLaVA-NEXT",
    "https://huggingface.co/liuhaotian/llava-v1.6-34b",
    "LLaVA-NEXT",
)

register_model_info(
    ["llava-v1.6-vicuna-13b"],
    "LLaVA-NEXT",
    "https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b",
    "LLaVA-NeXT-Vicuna-13B",
)

register_model_info(
    ["llava-v1.6-vicuna-7b"],
    "LLaVA-NEXT",
    "https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b",
    "LLaVA-NeXT-Vicuna-13B",
)

register_model_info(
    ["MiniCPM-V"],
    "MiniCPM-V-3B",
    "https://huggingface.co/openbmb/MiniCPM-V",
    "MiniCPM-V",
)

register_model_info(
    ["MiniCPM-V-2"],
    "MiniCPM-V-2",
    "https://huggingface.co/openbmb/MiniCPM-V-2",
    "MiniCPM-V-2",
)

register_model_info(
    ["minicpm-llama3-v"],
    "MiniCPM-Llama3-V-2.5",
    "https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5",
    "minicpm-llama3-v",
)

# https://huggingface.co/spaces/THUDM/CogVLM-CogAgent
register_model_info(
    ["cogvlm-chat-hf"],
    "CogVLM Chat",
    "https://huggingface.co/THUDM/cogvlm-chat-hf",
    "CogVLM Chat",
)

register_model_info(
    ["Qwen-VL-Chat"],
    "Qwen-VL-Chat",
    "https://huggingface.co/Qwen/Qwen-VL-Chat",
    "Qwen-VL-Chat",
)

register_model_info(
    ["Qwen-VL-Plus"],
    "Qwen-VL-Plus",
    "https://github.com/QwenLM/Qwen-VL",
    "Qwen-VL-Plus",
)

register_model_info(
    ["Qwen-VL-Max"],
    "Qwen-VL-Max",
    "https://github.com/QwenLM/Qwen-VL",
    "Qwen-VL-Max",
)

register_model_info(
    ["instructblip-vicuna-7b"],
    "InstructBLIP Vicuna 7b",
    "https://huggingface.co/Salesforce/instructblip-vicuna-7b",
    "InstructBLIP Vicuna 7b",
)

register_model_info(
    ["uform-gen2-qwen-500m"],
    "UForm-Gen2",
    "https://huggingface.co/unum-cloud/uform-gen2-qwen-500m",
    "UForm Gen2 Qwen 500M",
)

register_model_info(
    ["tiny-llava-v1-hf"],
    "TinyLLaVA",
    "https://huggingface.co/bczhou/tiny-llava-v1-hf",
    "TinyLLaVA",
)


register_model_info(
    ["Bunny-v1_0-3B"],
    "Bunny-v1_0-3B",
    "https://huggingface.co/BAAI/Bunny-v1_0-3B",
    "Bunny v1.0 3B",
)

register_model_info(
    ["claude-3-opus-20240229"],
    "claude-3-opus-20240229",
    "https://www.anthropic.com/claude",
    "Claude-3 Opus (20240229)",
)

register_model_info(
    ["claude-3-sonnet-20240229"],
    "claude-3-sonnet-20240229",
    "https://www.anthropic.com/claude",
    "Claude-3 Sonnet (20240229)",
)

register_model_info(
    ["claude-3-5-sonnet-20240620"],
    "claude-3-5-sonnet-20240620",
    "https://www.anthropic.com/claude",
    "Claude-3.5 Sonnet (20240620)",
)

register_model_info(
    ["claude-3-haiku-20240307"],
    "claude-3-haiku-20240307",
    "https://www.anthropic.com/claude",
    "Claude-3 Haiku (20240229)",
)

register_model_info(
    ["deepseek-vl-7b-chat"],
    "deepseek-vl-7b-chat",
    "https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat",
    "DeepSeek-VL-7B-Chat",
)

register_model_info(
    ["Yi-VL-6B"],
    "Yi-VL-6B",
    "https://huggingface.co/01-ai/Yi-VL-6B",
    "Yi-VL-6B",
)

register_model_info(
    ["yi-vl-plus"],
    "Yi-VL-Plus",
    "https://platform.lingyiwanwu.com/",
    "Yi-VL-Plus",
)

# register_model_info(
#     ["reka"],
#     "Reka",
#     "https://docs.reka.ai/index.html",
#     "Reka",
# )

# creeping-phlox-20240403
register_model_info(
    ["Reka-Flash"],
    "Reka Flash",
    "https://docs.reka.ai/index.html",
    "Reka Flash",
)

register_model_info(
    ["Reka-Core"],
    "Reka Core",
    "https://docs.reka.ai/index.html",
    "Reka Core",
)

# Register Idefics2 Local
register_model_info(
    ["idefics2-local"],
    "idefics2-8b-chatty",
    "https://huggingface.co/HuggingFaceM4/idefics2-8b",
    "Idefics2",
)

# Register Idefics2 API
register_model_info(
    ["idefics2-8b-chatty"],
    "idefics2-8b-chatty",
    "https://huggingface.co/HuggingFaceM4/idefics2-8b",
    "Idefics2 8b",
)

register_model_info(
    ["gpt-4-turbo"],
    "GPT-4-Turbo",
    "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
    "GPT-4-Turbo by OpenAI",
)

register_model_info(
    ["gpt-4-vision-preview"],
    "gpt-4-vision-preview",
    "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
    "GPT-4(V) by OpenAI",
)

register_model_info(
    ["gpt-4o"],
    "gpt-4o",
    "https://platform.openai.com/docs/models/gpt-4o",
    "GPT-4o by OpenAI",
)

# ======== VideoLLM ========

# Register Video-LLaVA
register_model_info(
    ["Video-LLaVA-7B"],
    "Video-LLaVA",
    "https://github.com/PKU-YuanGroup/Video-LLaVA",
    "Video-LLaVA by PKU",
)

# Register LITA
register_model_info(
    ["LITA-13B-v1.3"],
    "LITA",
    "https://github.com/NVlabs/LITA",
    "LITA by NVIDIA",
)