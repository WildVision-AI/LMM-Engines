import torch
from PIL import Image
import json
import base64
import os
import uuid
import queue
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image, decode_and_save_video
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from threading import Thread
from typing import List

Qwen2VL_MODEL_LIST = (
    "Qwen2-VL-2B-Instruct",
    "Qwen2-VL-7B-Instruct"
)


class GeneratorError(Exception):
    pass
class Qwen2VLAdapter(BaseModelAdapter):
    """The model adapter for Qwen2VL"""

    def match(self, model_path: str):
        return "qwen2-vl" in model_path.lower()
    
    def load_model(self, model_path: str, device: str, from_pretrained_kwargs: dict = ...):
        self.model = None
        if "torch_dtype" not in from_pretrained_kwargs:
            from_pretrained_kwargs["torch_dtype"] = torch.float16
        print(from_pretrained_kwargs)
        if not from_pretrained_kwargs.get("device_map"):
            from_pretrained_kwargs["device_map"] = "cuda"
        self.torch_dtype = from_pretrained_kwargs["torch_dtype"]
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **from_pretrained_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        return self.model
    
    def generate(self, params:dict):
        """
        params:dict = {
            "prompt": {
                "text": str,
                "video": str, # base64 encoded video
                "image": str, # base64 encoded imafe
            },
            **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
        }
        """
        prompt = params["prompt"]["text"]

        if "image" in params["prompt"].keys():
            os.makedirs("tmp_log_images", exist_ok=True)
            unique_filename = str(uuid.uuid4()) + '.png'
            tmp_img_save_path = os.path.join('tmp_log_images', unique_filename)
            im_b64 = json.loads(params["prompt"]["image"])
            im_bytes = base64.b64decode(im_b64)
            im_file = BytesIO(im_bytes)
            image = Image.open(im_file).convert('RGB')
            image_rgb = image.convert('RGB')
            image_rgb.save(tmp_img_save_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": tmp_img_save_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        elif "video" in params["prompt"].keys():
            video_path = decode_and_save_video(params["prompt"]["video"]) 
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]


        temperature = float(params.get("temperature", 0.2))
        top_p = float(params.get("top_p", 0.7))
        max_new_tokens = min(int(params.get("max_new_tokens", 1024)), 1024)
        do_sample = temperature > 0.0
        temperature = None if not do_sample else temperature
   
        from icecream import ic
        ic(prompt)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        try:
            os.remove(tmp_img_save_path)
        except:
            pass

        return {"text": output_text}
    
    
    def generate_stream(self, params:dict):
        """
        params:dict = {
            "prompt": {
                "text": str,
                "video": str, # base64 encoded video
                "image": str, # base64 encoded imafe
            },
            **generation_kwargs # other generation kwargs, like temperature, top_p, max_new_tokens, etc.
        }
        """
        prompt = params["prompt"]["text"]

        if "image" in params["prompt"].keys():
            os.makedirs("tmp_log_images", exist_ok=True)
            unique_filename = str(uuid.uuid4()) + '.png'
            tmp_img_save_path = os.path.join('tmp_log_images', unique_filename)
            im_b64 = json.loads(params["prompt"]["image"])
            im_bytes = base64.b64decode(im_b64)
            im_file = BytesIO(im_bytes)
            image = Image.open(im_file).convert('RGB')
            image_rgb = image.convert('RGB')
            image_rgb.save(tmp_img_save_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": tmp_img_save_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        elif "video" in params["prompt"].keys():
            video_path = decode_and_save_video(params["prompt"]["video"]) 
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]


        temperature = float(params.get("temperature", 0.2))
        top_p = float(params.get("top_p", 0.7))
        max_new_tokens = min(int(params.get("max_new_tokens", 1024)), 1024)
        do_sample = temperature > 0.0
        temperature = None if not do_sample else temperature
   
        from icecream import ic
        ic(prompt)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        tokenizer = self.processor.tokenizer
        streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            'max_new_tokens': max_new_tokens, 
            'streamer': streamer, 
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': do_sample,
            **inputs}

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ''
        for new_text in streamer:
            generated_text += new_text
            yield {"text": generated_text}
        thread.join()
        try:
            os.remove(tmp_img_save_path)
        except:
            pass

    def get_info(self):

        return {
            "type": "image;video",
            "author": "Anonymous",
            "organization": "Anonymous",
            "model_size": None,
            "model_link": None,
        }
    


if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    device = "cuda"
    model_adapter = Qwen2VLAdapter()
    test_adapter(model_adapter, model_path, device)
"""
python -m lmm_engines.huggingface.model.model_qwen2vl

bash start_worker_on_arena.sh Qwen/Qwen2-VL-7B-Instruct 41414
"""