import json
from io import BytesIO
from .model_adapter import BaseModelAdapter
from ...utils import decode_image, encode_image
from typing import List
from PIL import Image
from ...vllm import launch_vllm_worker, call_vllm_worker
import base64
from PIL import Image

class VLLMAriaAdapter(BaseModelAdapter):
    """The model adapter for Aria"""
    

    def match(self, model_path: str):
        return "vllm" in model_path.lower() and "aria" in model_path.lower()

    def load_model(self, model_path: str, device:str="cuda", from_pretrained_kwargs: dict={}):
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
        assert model_path.startswith("vllm/")
        self.model_name = model_path[len("vllm/"):]
        max_img_per_msg = 256
        worker_addr, proc = launch_vllm_worker(
            self.model_name,
            num_gpus=1,
            additional_args=[
                "--max-model-len", "38400", "--tokenizer", self.model_name,
                "--limit-mm-per-prompt", f"image={max_img_per_msg}",
                "--enforce-eager", "--trust-remote-code", "--gpu-memory-utilization", "0.84"
                ],
        )
        self.worker_addr = worker_addr
        self.proc = proc
        self.model = None
        return worker_addr
    
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
        # add your custom generation code here
        image = decode_image(params["prompt"]["image"]) # This image will be decoded into a PIL image
        # image = image.resize((3844, 2408))
        image_url = f"data:image/jpeg;base64,{json.loads(encode_image(image, image_format='JPEG'))}"
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        if "frame_num" in generation_kwargs:
            frame_num = generation_kwargs.pop("frame_num")
        else:
            frame_num = 32
        vllm_generation_kwargs = {
            "temperature": generation_kwargs.get("temperature", 0.0),
            "top_p": generation_kwargs.get("top_p", 1.0),
            "max_tokens": generation_kwargs.get("max_new_tokens", 8192),
        }
        
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
            },
        ]
        response = call_vllm_worker(
            messages=messages,
            worker_addrs=[self.worker_addr],
            model_name=self.model_name,
            timeout=300,
            **vllm_generation_kwargs
        )
        return {"text": response}
        
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
        image = decode_image(params["prompt"]["image"]) # This image will be decoded into a PIL image
        # image = image.resize((3844, 2408)) # for the image shape, otherwise bug
        image_url = f"data:image/jpeg;base64,{json.loads(encode_image(image, image_format='JPEG'))}"
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        if "frame_num" in generation_kwargs:
            frame_num = generation_kwargs.pop("frame_num")
        else:
            frame_num = 32
        vllm_generation_kwargs = {
            "temperature": generation_kwargs.get("temperature", 0.0),
            "top_p": generation_kwargs.get("top_p", 1.0),
            "max_tokens": generation_kwargs.get("max_new_tokens", 8192),
            "stream": True,
        }
        
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
            },
        ]
        response = call_vllm_worker(
            messages=messages,
            worker_addrs=[self.worker_addr],
            model_name=self.model_name,
            **vllm_generation_kwargs
        )
        generated_text = ""
        
        for text in response:
            generated_text += text
            yield {"text": generated_text}
    
    def get_info(self):
        return {
            "type": "image",
            "author": "RhymesAI",
            "organization": "RhymesAI",
            "model_size": "25.3B",
            "model_link": "https://huggingface.co/rhymes-ai/Aria"
        }
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "vllm/rhymes-ai/Aria"
    device = "cuda"
    model_adapter = VLLMAriaAdapter()
    test_adapter(model_adapter, model_path, device)
    
"""
# local testing
python -m lmm_engines.huggingface.model.model_vllm_aria
# connect to wildvision arena
bash start_worker_on_arena.sh vllm/rhymes-ai/Aria 41412
"""