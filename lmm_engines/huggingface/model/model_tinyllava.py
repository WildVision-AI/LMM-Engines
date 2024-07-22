import torch
from PIL import Image
import json
import base64
from io import BytesIO
from .model_adapter import BaseModelAdapter, register_model_adapter
from ..conversation import get_conv_template, Conversation
from ...utils import decode_image
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer, LlavaForConditionalGeneration
from threading import Thread
from typing import List

class TinyLLaVAAdapter(BaseModelAdapter):
    """The model adapter for TinyLLaVA"""

    def match(self, model_path: str):
        return "tiny-llava" in model_path.lower()

    def load_model(self, model_path: str, device:str="cuda", from_pretrained_kwargs: dict={}):
        model_id = model_path
        if "torch_dtype" not in from_pretrained_kwargs:
            from_pretrained_kwargs["torch_dtype"] = torch.float16
        self.torch_dtype = from_pretrained_kwargs["torch_dtype"]
        from_pretrained_kwargs["device_map"] = device
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, **from_pretrained_kwargs
        )
        return self.model
    
    def generate(self, params:List[dict]):
        
        # extract params
        prompt = params["prompt"]["text"]
        image = decode_image(params["prompt"]["image"])
        
        # format generation kwargs
        default_generation_kwargs = {"max_new_tokens": 200, "temperature": 0.2, "top_p": 0.7}
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        default_generation_kwargs.update(generation_kwargs)
        generation_kwargs = default_generation_kwargs
        generation_kwargs["do_sample"] = generation_kwargs["temperature"] > 0.0
        
        # format prompt 
        template = "USER: <image>\n{prompt}\nASSISTANT:"
        prompt = prompt if "ASSISTANT:" in prompt else template.format(prompt=prompt)
        
        # generation
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.device, self.torch_dtype)
        outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_text = self.processor.decode(outputs[0][2:], skip_special_tokens=True)
        index = generated_text.rfind("ASSISTANT:")
        generated_text = generated_text[index+len("ASSISTANT:")+1:].lstrip() if index > -1 else generated_text
        return {"text": generated_text}
        
    def generate_stream(self, params:List[dict]):
        
        # extract params
        prompt = params["prompt"]["text"]
        image = decode_image(params["prompt"]["image"])
        
        # format generation kwargs
        default_generation_kwargs = {"max_new_tokens": 200, "temperature": 0.2, "top_p": 0.7}
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        default_generation_kwargs.update(generation_kwargs)
        generation_kwargs = default_generation_kwargs
        generation_kwargs["do_sample"] = generation_kwargs["temperature"] > 0.0
        
        # add streamer
        streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        
        # format prompt 
        template = "USER: <image>\n{prompt}\nASSISTANT:"
        prompt = prompt if "ASSISTANT:" in prompt else template.format(prompt=prompt)
        
        # generation
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.device, self.torch_dtype)
        thread = Thread(target=self.model.generate, kwargs={**inputs, **generation_kwargs})
        thread.start()
        
        generated_text = ""
        is_assistant_removed = False
        for text in streamer:
            generated_text += text
            if len(generated_text) < len("ASSISTANT:") + 2:
                continue
            else:
                if not is_assistant_removed:
                    index = generated_text.rfind("ASSISTANT:")
                    if index > -1:
                        generated_text = generated_text[index+len("ASSISTANT:")+1:].lstrip()
                    is_assistant_removed = True
                yield {"text": generated_text}

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("tiny-llava")


@torch.inference_mode()
def generate_stream_tinyllava(model, tokenizer, image_processor, params, device, context_len, stream_interval, judge_sent_end=False):
    pipe = model
    
    prompt = params["prompt"]["text"]
    im_b64 = json.loads(params["prompt"]["image"])
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file)
    
    temperature = float(params.get("temperature", 0.2))
    top_p = float(params.get("top_p", 0.7))
    do_sample = temperature > 0.0
    max_new_tokens = min(int(params.get("max_new_tokens", 200)), 200)
    
    template = "USER: <image>\n{prompt}\nASSISTANT:"
    prompt = prompt if "ASSISTANT:" in prompt else template.format(prompt=prompt)
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "do_sample": do_sample})
    index = outputs[0]["generated_text"].rfind("ASSISTANT:")
    generated_text = outputs[0]["generated_text"][index+len("ASSISTANT:")+1:]
    yield {"text": generated_text}
    
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "bczhou/tiny-llava-v1-hf"
    device = "cuda:0"
    from_pretrained_kwargs = {"torch_dtype": torch.float16}
    model_adapter = TinyLLaVAAdapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter)
    
"""
python -m lmm_engines.huggingface.model.model_tinyllava
"""