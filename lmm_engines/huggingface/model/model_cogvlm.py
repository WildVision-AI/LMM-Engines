
import torch
from PIL import Image
import json
import base64
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image, decode_and_save_video
from .videollm_utils.cogvlm.utils import load_video
from transformers import CLIPImageProcessor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from threading import Thread
from typing import List

class CogVLMAdapter(BaseModelAdapter):
    """The model adapter for CogVLM"""

    def match(self, model_path: str):
        return "cogvlm" in model_path.lower() and "cogvlm2-video-llama3-chat" not in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("cogvlm")
    
    def load_model(self, model_path: str, device: str, from_pretrained_kwargs: dict = ...):
        pass
    
    def generate(self, params:dict):
        pass
        
    def generate_stream(self, params:dict):
        pass
    
class CogVLM2VideoAdapter(BaseModelAdapter):
    """The model adapter for CogVLM2Video"""

    def match(self, model_path: str):
        return "cogvlm2-video-llama3-chat" in model_path.lower()

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
        # from_pretrained_kwargs["torch_dtype"] = torch.float16
        self.torch_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        from_pretrained_kwargs["torch_dtype"] = self.torch_type
        from_pretrained_kwargs["low_cpu_mem_usage"] = True
        from_pretrained_kwargs["trust_remote_code"] = True
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, **from_pretrained_kwargs
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
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
        strategy = "chat"
        video = load_video(video_path, strategy=strategy)

        history = []
        inputs = self.model.build_conversation_input_ids(
            tokenizer = self.tokenizer,
            query = prompt,
            images = [video],
            history = history,
            template_version = strategy
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to(self.model.device).to(self.torch_type)]],
        }
        # gen_kwargs = {
        # "max_new_tokens": 2048,
        # "pad_token_id": 128002,
        # "top_k": 1,
        # "do_sample": False,
        # "top_p": 0.1,
        # "temperature": temperature,
        # }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        strategy = "chat"
        video = load_video(video_path, strategy=strategy)

        history = []
        inputs = self.model.build_conversation_input_ids(
            tokenizer = self.tokenizer,
            query = prompt,
            images = [video],
            history = history,
            template_version = strategy
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to(self.model.device).to(self.torch_type)]],
        }

        # add streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer

        # Start the model chat in a separate thread
        thread = Thread(target=self.model.generate, kwargs={**inputs, **generation_kwargs})
        thread.start()
        
        generated_text = ""
        for text in streamer:
            generated_text += text
            yield {"text": generated_text}
            
    def get_info(self):
        return {
            "type": "video",
            "author": "Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang",
            "organization": "THUDM",
            "model_size": "8B",
            "model_link": "https://huggingface.co/THUDM/cogvlm2-video-llama3-chat",
        }


if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    # model_path = "..."
    # device = "cuda"
    # model_adapter = CogVLMAdapter()
    # test_adapter(model_adapter, model_path, device)

    model_path = "THUDM/cogvlm2-video-llama3-chat"
    device = "cuda"
    model_adapter = CogVLM2VideoAdapter()
    test_adapter(model_adapter, model_path, device)
"""
python -m lmm_engines.huggingface.model.model_cogvlm
# connect to wildvision arena
bash start_worker_on_arena.sh THUDM/cogvlm2-video-llama3-chat 41417 1
"""