
import torch
from PIL import Image
import json
import os
import av
import numpy as np
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_and_save_video
from transformers import TextIteratorStreamer, VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from threading import Thread
from typing import List
from huggingface_hub import hf_hub_download


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
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

class VideoLLaVAAdapter(BaseModelAdapter):
    """The model adapter for LanguageBind/Video-LLaVA-7B"""

    def match(self, model_path: str):
        return "video-llava" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("video-llava")
    
    def load_model(self, model_path: str, device: str, from_pretrained_kwargs: dict = ...):
        """
        load all the elements of the models here that will be used for your model's geneation, such as the model, tokenizer, processor, etc.
        Args:
            model_path (str): the path to the model, huggingface model id or local path
            device (str): the device to run the model on. e.g. "cuda" or "cpu", it cannot be used to load a model, use device_map in from_pretrained_kwargs instead.
            from_pretrained_kwargs (dict): other kwargs to pass to the from_pretrained method.
                including device_map, torch_dtype, etc.
                we use device_map so that we can run the model on multiple devices
        Returns:
            model: A nn.Module model or huggingface PreTrainedModel model
        """
        self.model = None
        if "torch_dtype" not in from_pretrained_kwargs:
            from_pretrained_kwargs["torch_dtype"] = torch.float16
        print(from_pretrained_kwargs)
        if not from_pretrained_kwargs.get("device_map"):
            from_pretrained_kwargs["device_map"] = "cuda"
        self.torch_dtype = from_pretrained_kwargs["torch_dtype"]
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(model_path, **from_pretrained_kwargs)
        self.processor = VideoLlavaProcessor.from_pretrained(model_path)
        
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
        
        container = av.open(video_path)

        # sample uniformly 8 frames from the video
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)

        final_prompt = f"USER: <video>{prompt} ASSISTANT:"
        inputs = self.processor(text=final_prompt, videos=clip, return_tensors="pt")

        # Generate
        generate_ids = self.model.generate(**inputs, **generation_kwargs)
        input_len = inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generate_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
        # add your custom generation code here
        video_path = decode_and_save_video(params["prompt"]["video"]) # This will save the video to a file and return the path
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        # add streamer
        streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        
        container = av.open(video_path)

        # sample uniformly 8 frames from the video
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)

        final_prompt = f"USER: <video>{prompt} ASSISTANT:"
        inputs = self.processor(text=final_prompt, videos=clip, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        thread = Thread(target=self.model.generate, kwargs={**inputs, **generation_kwargs})
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
    model_path = "LanguageBind/Video-LLaVA-7B-hf"
    device = "cuda:0"
    from_pretrained_kwargs = {"torch_dtype": torch.float16}
    model_adapter = VideoLLaVAAdapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter, model_type="video")
    
"""
python -m lmm_engines.huggingface.model.model_videollava
# connect to wildvision arena
bash start_worker_on_arena.sh LanguageBind/Video-LLaVA-7B-hf 41411 1
"""