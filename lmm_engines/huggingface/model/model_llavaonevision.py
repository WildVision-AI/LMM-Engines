import torch
from PIL import Image
import json
import time
import av
import numpy as np
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_and_save_video
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, TextIteratorStreamer
from transformers.utils import is_flash_attn_2_available
from threading import Thread
from typing import List

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

class LLaVAOnevisionAdapter(BaseModelAdapter):
    """The model adapter for LLaVAOneVision"""

    def match(self, model_path: str):
        return "LLaVA-Onevision".lower() in model_path.lower()

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
        if "torch_dtype" not in from_pretrained_kwargs:
            from_pretrained_kwargs["torch_dtype"] = torch.float16
        from_pretrained_kwargs["low_cpu_mem_usage"] = True
        if is_flash_attn_2_available():
            from_pretrained_kwargs["use_flash_attention_2"] = True
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, **from_pretrained_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        
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
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video"},
                    ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        container = av.open(video_path)

        # sample uniformly 8 frames from the video, can sample more for longer videos
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)
        inputs_video = self.processor(text=prompt, videos=list(clip), padding=True, return_tensors="pt").to(self.model.device, torch.float16)
        
        output = self.model.generate(**inputs_video, **generation_kwargs)
        input_len = inputs_video["input_ids"].shape[1]
        generated_text = self.processor.decode(output[0][input_len:], skip_special_tokens=True)
        return {"text": generated_text}
        
    def generate_stream(self, params:dict):
        """
        params:dict = {
            "prompt": {
                "text": str,
                "image": str, # base64 image
            },
            **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
        }
        """
        video_path = decode_and_save_video(params["prompt"]["video"]) # This will save the video to a file and return the path
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video"},
                    ],
            },
        ]

        # add streamer
        streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        container = av.open(video_path)

        # sample uniformly 8 frames from the video, can sample more for longer videos
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)
        inputs_video = self.processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(self.model.device, torch.float16)
        
        thread = Thread(target=self.model.generate, kwargs={**inputs_video, **generation_kwargs})
        thread.start()
        
        generated_text = ""
        for text in streamer:
            generated_text += text
            yield {"text": generated_text}
            
    def get_info(self):
        return {
            "type": "video",
            "author": "Anonymous",
            "organization": "Anonymous",
            "model_size": None,
            "model_link": None,
        }
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    device = "cuda"
    model_adapter = LLaVAOnevisionAdapter()
    test_adapter(model_adapter, model_path, device)
    
"""
# local testing
python -m lmm_engines.huggingface.model.model_llavaonevision
"""