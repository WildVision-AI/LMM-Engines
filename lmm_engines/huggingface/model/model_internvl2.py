import torch
from PIL import Image
import json
import time
import math
import numpy as np
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from .videollm_utils.internvl2.utils import load_video
from ...utils import decode_and_save_video
from transformers import TextIteratorStreamer, AutoModel, AutoProcessor, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from threading import Thread
from typing import List

InternVL2_MODEL_LIST = (
    "InternVL2‑1B",
    "InternVL2‑2B",
    "InternVL2‑4B",
    "InternVL2‑8B",
    "InternVL2‑26B",
    "InternVL2‑40B",
    "InternVL2-Llama3-76B",
)

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

class InternVL2Adapter(BaseModelAdapter):
    """The model adapter for InternVL2"""

    def match(self, model_path: str):
        return "InternVL2".lower() in model_path.lower()

    def load_model(self, model_path: str, device:str="cuda", from_pretrained_kwargs: dict={}):
        """
        load all the elements of the models here that will be used for your model's geneation, such as the model, tokenizer, processor, etc.
        Args:
            model_path (str): the path to the model, huggingface model id or local path
            device (str): the device to run the model on
            from_pretrained_kwargs (dict): other kwargs to pass to the from_pretrained method.
                It's better to ignore this one, and set your custom kwargs in the load_model method.
        Returns:
            model: A nn.Module model or huggingface PreTrainedModel model
        """
        from_pretrained_kwargs["torch_dtype"] = torch.bfloat16
        from_pretrained_kwargs["low_cpu_mem_usage"] = True
        from_pretrained_kwargs["trust_remote_code"] = True
        if is_flash_attn_2_available():
            from_pretrained_kwargs["use_flash_attn"] = True
        model_name = model_path.split("/")[1]
        if model_name in ["InternVL2-40B", "InternVL2-Llama3-76B"]:
            device_map = split_model(model_name)
            from_pretrained_kwargs["device_map"] = device_map
        self.model = AutoModel.from_pretrained(
            model_path, **from_pretrained_kwargs
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        
        return self.model
    
    def generate(self, params:dict):
        """
        generation
        Args:
            params:dict = {
                "prompt": {
                    "text": str,
                    "video": str, # base64 encoded video
                },
                **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
            }
        Returns:
            {"text": ...}
        """
        # add your custom generation code here
        video_path = decode_and_save_video(params["prompt"]["video"]) # This will save the video to a file and return the path
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + prompt
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
        return {"text": response}
        
    def generate_stream(self, params:dict):
        """
        params:dict = {
            "prompt": {
                "text": str,
                "video": str, # base64 encoded video
            },
            **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
        }
        """
        video_path = decode_and_save_video(params["prompt"]["video"]) # This will save the video to a file and return the path
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + prompt

        # add streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
        generation_config = dict(max_new_tokens=1024, do_sample=False, streamer=streamer)

        # Start the model chat in a separate thread
        thread = Thread(target=self.model.chat, kwargs=dict(
            tokenizer=self.tokenizer, pixel_values=pixel_values, question=question,
            history=None, return_history=False, generation_config=generation_config,
        ))
        thread.start()
        
        generated_text = ""
        for text in streamer:
            generated_text += text
            yield {"text": generated_text}
            
    def get_info(self):
        return {
            "type": "video",
            "author": "Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng",
            "organization": "OpenGVLab",
            "model_size": "8B",
            "model_link": "https://huggingface.co/OpenGVLab/InternVL2-8B",
        }
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "OpenGVLab/InternVL2-8B"
    # model_path = "OpenGVLab/InternVL2-Llama3-76B"
    device = "cuda"
    num_gpus = torch.cuda.device_count()
    model_adapter = InternVL2Adapter()
    test_adapter(model_adapter, model_path, device, num_gpus=num_gpus)
    
"""
# local testing
python -m lmm_engines.huggingface.model.model_internvl2
# connect to wildvision arena
bash start_worker_on_arena.sh OpenGVLab/InternVL2-8B 42003 1
"""