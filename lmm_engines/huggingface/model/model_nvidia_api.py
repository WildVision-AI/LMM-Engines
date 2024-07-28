import torch
from PIL import Image
import json
import base64
import requests
import time
import io
import os
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer, LlavaForConditionalGeneration
from threading import Thread
from typing import List


def convert_response_to_text(response, stream):

    if not stream:
        full_content = response.json()['choices'][0]['message']['content']
    else:
        contents = []
        for line in response.iter_lines():
            # Decode the line to a string
            line = line.decode('utf-8')
            # Remove the 'data: ' prefix if present
            if line.startswith('data: '):
                line = line[len('data: '):]
            # Ignore empty lines and the '[DONE]' line
            if not line.strip() or line.strip() == '[DONE]':
                continue
            # Parse the JSON data
            data = json.loads(line)
            # Extract the content field
            content = data['choices'][0]['delta'].get('content', '')
            # Append the content to the list
            contents.append(content)

        # Concatenate all the content into a single string
        full_content = ''.join(contents)

    return full_content


class NvidiaAPIAdapter(BaseModelAdapter):
    """The model adapter for Nvidia API"""

    def __init__(self) -> None:
        super().__init__()
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.invoke_url = "https://ai.api.nvidia.com/v1/vlm/community/llava16-34b"
    

    def match(self, model_path: str):
        return "nvidia_api" in model_path.lower()

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
        self.model = model_path
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set")
        print(self.model)
        return self.model
    
    def generate(self, params:dict):
        """
        generation
        Args:
            params:dict = {
                "prompt": {
                    "text": str,
                    "image": str, # base64 image
                },
                **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
            }
        Returns:
            {"text": ...}
        """
        stream = False

        image = decode_image(params["prompt"]["image"]) # This image will be decoded into a PIL image
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  # Change format if your image is in a different format (e.g., "PNG")
        # Encode byte array to base64
        image_b64 = base64.b64encode(buffered.getvalue()).decode()
        prompt = params["prompt"]["text"]

        headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Accept": "text/event-stream" if stream else "application/json"
        }

        payload = {
        "messages": [
            {
            "role": "user",
            "content": f'{prompt} <img src="data:image/jpeg;base64,{image_b64}" />'
            }
        ],
        "max_tokens": params.get("max_new_tokens", params.get("max_tokens", 512)),
        "temperature": params.get("temperature", 0.0),
        "top_p": params.get("top_p", 1.0),
        "stream": stream
        }

        response = requests.post(self.invoke_url, headers=headers, json=payload)
        text = convert_response_to_text(response, stream)

        return {"text": text}
        
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
        stream = True

        image = decode_image(params["prompt"]["image"]) # This image will be decoded into a PIL image
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  # Change format if your image is in a different format (e.g., "PNG")
        # Encode byte array to base64
        image_b64 = base64.b64encode(buffered.getvalue()).decode()
        prompt = params["prompt"]["text"]

        headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Accept": "text/event-stream" if stream else "application/json"
        }

        payload = {
        "messages": [
            {
            "role": "user",
            "content": f'{prompt} <img src="data:image/jpeg;base64,{image_b64}" />'
            }
        ],
        "max_tokens": params.get("max_new_tokens", params.get("max_tokens", 512)),
        "temperature": params.get("temperature", 0.0),
        "top_p": params.get("top_p", 1.0),
        "stream": stream
        }

        response = requests.post(self.invoke_url, headers=headers, json=payload, stream=stream)

        # Check if the response is streamed
        if response.status_code == 200:
            # Iterate over the response lines as they arrive
            generated_text = ""
            for line in response.iter_lines():
                if line:
                    # Decode the line from bytes to string
                    decoded_line = line.decode("utf-8")
                    # Remove the 'data: ' prefix if present
                    if decoded_line.startswith('data: '):
                        decoded_line = decoded_line[len('data: '):]
                    # Ignore empty lines and the '[DONE]' line
                    if not decoded_line.strip() or decoded_line.strip() == '[DONE]':
                        continue
                    # Parse the JSON data
                    data = json.loads(decoded_line)
                    # Extract the content field
                    content = data['choices'][0]['delta'].get('content', '')
                    # Print or process the content
                    generated_text += content
                    yield {"text": generated_text}
        else:
            print("Failed to connect, status code:", response.status_code)

    
    def get_info(self):
        return {
            "type": "image",
            "author": "Anonymous",
            "organization": "Anonymous",
            "model_size": None,
            "model_link": None,
        }
        
        
class LLaVAv1634bNvidiaAPIAdapter(NvidiaAPIAdapter):
    """The model adapter for LLaVA v1.6 34b Nvidia API"""

    def __init__(self) -> None:
        super().__init__()
        self.invoke_url = "https://ai.api.nvidia.com/v1/vlm/community/llava16-34b"
    
    def match(self, model_path: str):
        return "llava_v1_6_34b_nvidia_api" in model_path.lower()
    
    def get_info(self):
        return {
            "type": "image",
            "author": "...",
            "organization": "...",
            "model_size": "34b",
            "model_link": "https://build.nvidia.com/explore/vision#llava16-34b"
        }
    
class LLaVAv16Mistral7bNvidiaAPIAdapter(NvidiaAPIAdapter):
    """The model adapter for LLaVA v1.6 Mistral 7b Nvidia API"""

    def __init__(self) -> None:
        super().__init__()
        self.invoke_url = "https://ai.api.nvidia.com/v1/vlm/community/llava16-mistral-7b"
    
    def match(self, model_path: str):
        return "llava_v1_6_mistral_7b_nvidia_api" in model_path.lower()
    
    def get_info(self):
        return {
            "type": "image",
            "author": "...",
            "organization": "...",
            "model_size": "7b",
            "model_link": "https://build.nvidia.com/explore/vision#llava16-mistral-7b"
        }
        
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    device = "cuda:0"
    from_pretrained_kwargs = {"torch_dtype": torch.float16}
    
    model_path = "llava_v1_6_34b_nvidia_api"
    model_adapter = LLaVAv1634bNvidiaAPIAdapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter)
    
    model_path = "llava_v1_6_mistral_7b_nvidia_api"
    model_adapter = LLaVAv16Mistral7bNvidiaAPIAdapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter)
    
"""
export NVIDIA_API_KEY={your_nvidia_api_key} # see https://build.nvidia.com/explore/vision
# local testing
python -m lmm_engines.huggingface.model.model_nvidia_api
# connect to wildvision arena
bash start_worker_on_arena.sh llava_v1_6_34b_nvidia_api 41410
bash start_worker_on_arena.sh llava_v1_6_mistral_7b_nvidia_api 41411
"""