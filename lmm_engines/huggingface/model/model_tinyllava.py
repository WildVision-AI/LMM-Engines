import torch
from PIL import Image
import json
import base64
from io import BytesIO
from icecream import ic
from .model_adapter import BaseModelAdapter
from ..conversation import get_conv_template, Conversation
from transformers import AutoTokenize, AutoModel, AutoProcessor, pipeline, TextStreamer, LlavaForConditionalGeneration
from typing import List


class TinyLLaVAAdapter(BaseModelAdapter):
    """The model adapter for TinyLLaVA"""

    def match(self, model_path: str):
        return "tiny-llava" in model_path.lower()

    def load_model(self, model_path: str, device:str, from_pretrained_kwargs: dict):
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
        
        # pipe = pipeline("image-to-text", model=model_id, device=device)
        # self.pipe = pipe
    
    def generate(self, params:List[dict]):
        
        # extract params
        prompt = params["prompt"]["text"]
        im_b64 = json.loads(params["prompt"]["image"])
        im_bytes = base64.b64decode(im_b64)
        im_file = BytesIO(im_bytes)
        image = Image.open(im_file)
        
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
        
        yield {"text": generated_text}
        
    def generate_stream(self, params:List[dict]):
        streamer = TextStreamer(self.processor)
        params["streamer"] = streamer
        generator = self.generate(params)["text"]
        
        generated_text = ""
        for text in generator:
            generated_text += text
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