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
from ...utils import encode_image, decode_image, decode_and_save_video, get_vision_input
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer, LlavaForConditionalGeneration
from threading import Thread
from typing import List

Gemini_MODEL_LIST = (
    "gemini-pro", 
    "gemini-pro-vision", 
    "gemini-1.5-flash-latest", 
    "gemini-1.5-pro-latest"
)

import google.generativeai as genai

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

class GeminiAdapter(BaseModelAdapter):
    """The model adapter for Gemini"""

    def match(self, model_path: str):
        return model_path in Gemini_MODEL_LIST

    def load_model(self, model_path: str, device:str="cuda", from_pretrained_kwargs: dict={}, model_type:str="image"):

        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)

        self.model_name = model_path
        self.model_type = model_type
        self.model = genai.GenerativeModel(model_path)

        return self.model

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bard")
    
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

        video_file = None
        if "video" in params['prompt'].keys():
            video_path = decode_and_save_video(params["prompt"]["video"])
            video_file = genai.upload_file(path=video_path)
            while video_file.state.name == "PROCESSING":
                print('Waiting for video to be processed.')
                import time
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)
        else:
            vision_input = decode_image(params["prompt"]["image"])
            image_list = get_vision_input(vision_input)

        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")

        if video_file:
            content = [prompt, video_file]
        else:
            content = [prompt, ]
            for image_pil in image_list:
                content.append(decode_image(encode_image(image_pil)))

        response = self.model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=min(int(generation_kwargs.get("max_new_tokens", 1024)), 1024),
                temperature=float(generation_kwargs.get("temperature", 0.2)),
                top_p = float(generation_kwargs.get("top_p", 0.7)),
            )
        )
        response.resolve()
        return {"text": response.text}
    
    def generate_image(self, params:dict):
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
    model_path = "gemini-1.5-pro-latest"
    device = "cuda"
    model_adapter = GeminiAdapter()
    test_adapter(model_adapter, model_path, device)
"""
python -m lmm_engines.huggingface.model.model_gemini
"""