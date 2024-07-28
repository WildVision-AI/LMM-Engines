import torch
from PIL import Image
import json
import base64
import requests
import time
import io
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


class LLaVAv1634bNvidiaAPIAdapter(BaseModelAdapter):
    """The model adapter for Nvidia API"""

    def __init__(self) -> None:
        super().__init__()
        self.api_key = "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"
        self.invoke_url = "https://ai.api.nvidia.com/v1/vlm/community/llava16-34b"
    

    def match(self, model_path: str):
        return "api" in model_path.lower()

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
        "max_tokens": 512,
        "temperature": 1.00,
        "top_p": 0.70,
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
        "max_tokens": 512,
        "temperature": 1.00,
        "top_p": 0.70,
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
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "llava_v1_6_34b_nvidia_api"
    device = "cuda:0"
    from_pretrained_kwargs = {"torch_dtype": torch.float16}
    model_adapter = LLaVAv1634bNvidiaAPIAdapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter)
    
"""
# local testing
python -m lmm_engines.huggingface.model.model_llava_v1_6_34b_nvidia_api
"""