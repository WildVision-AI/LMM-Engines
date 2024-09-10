import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import json
import base64
from io import BytesIO
import av
import numpy as np
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image, decode_and_save_video, get_vision_input, convert_pil_to_base64
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer, LlavaForConditionalGeneration
from threading import Thread
from typing import List

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-2024-05-13"
)

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




class OpenAIAdapter(BaseModelAdapter):
    """The model adapter for ChatGPT"""

    def match(self, model_path: str):
        return model_path in OPENAI_MODEL_LIST

    def load_model(self, model_path: str, device:str="cuda", from_pretrained_kwargs: dict={}, model_type:str="image"):
        # raise NotImplementedError()
        from openai import OpenAI
        
        api_key, api_base = None, None
        if "api_key" in from_pretrained_kwargs.keys():
            api_key = from_pretrained_kwargs

        self.model_name = model_path
        self.model_type = model_type

        self.model = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            base_url=api_base or "https://api.openai.com/v1"
        )

        return self.model

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("chatgpt")
    
    def generate(self, params:dict):
        """
        generation
        Args:
            params:dict = {
                "prompt": {
                    "text": str,
                    "image": str, # base64 image
                    "video": str, # base64 encoded video
                },
                **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
            }
        Returns:
            {"text": ...}
        """

        if "video" in params['prompt'].keys():
            video_path = decode_and_save_video(params["prompt"]["video"])
            container = av.open(video_path)

            # sample uniformly 8 frames from the video
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / params['num_frames']).astype(int)
            vision_input = read_video_pyav(container, indices)
        else:
            vision_input = decode_image(params["prompt"]["image"])

        image_list = get_vision_input(vision_input)

        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")

        input_messages = [{"role": "user", "content": None}]
        input_messages[-1]["content"] = [
            {"type": "text", "text": prompt},
        ]
        if image_list:
            for image_pil in image_list:
                base64_image = convert_pil_to_base64(image_pil)
                input_messages[-1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "auto"
                    },},)

 
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=input_messages,
            max_tokens=min(int(generation_kwargs.get("max_new_tokens", 1024)), 1024),
            temperature=float(generation_kwargs.get("temperature", 0.2)),
            top_p = float(generation_kwargs.get("top_p", 0.7)),
        )

        return {"text": response.choices[0].message.content}

    
    def generate_stream(self, params:dict):
        pass

    def get_info(self):

        return {
            "type": self.model_type,
            "author": "Anonymous",
            "organization": "Anonymous",
            "model_size": None,
            "model_link": None,
        }
    
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "gpt-4o-2024-05-13"
    device = "cuda"
    model_adapter = OpenAIAdapter()
    test_adapter(model_adapter, model_path, device)
"""
python -m lmm_engines.huggingface.model.model_openai
"""