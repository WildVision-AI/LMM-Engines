

import torch
from PIL import Image
import json
import base64
import time
import cv2
import os
import numpy as np
from io import BytesIO
from .model_adapter import BaseModelAdapter
from ...utils import decode_and_save_video
from threading import Thread
from typing import List
from decord import VideoReader, cpu
from transformers import TextIteratorStreamer
from .vlm_utils.llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .vlm_utils.llavavid.conversation import conv_templates, SeparatorStyle
from .vlm_utils.llavavid.model.builder import load_pretrained_model
from .vlm_utils.llavavid.utils import disable_torch_init
from .vlm_utils.llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def load_video(video_path, for_get_frames_num, force_sample=False):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    if len(frame_idx) > for_get_frames_num or force_sample:
        sample_fps = for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames


def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames


class LLaVANextVideoQwenAdapter(BaseModelAdapter):
    """The model adapter for LLaVA-NeXT-Video-32B-Qwen"""

    def match(self, model_path: str):
        return "LLaVA-NeXT-Video-32B-Qwen".lower() in model_path.lower()

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
        model_name = get_model_name_from_path(model_path)
        load_kwargs = {
            "device_map": from_pretrained_kwargs.get("device_map", "cuda"),
            "load_4bit": from_pretrained_kwargs.get("load_in_4bit", False),
            "load_8bit": from_pretrained_kwargs.get("load_in_8bit", False),
            "max_memory": from_pretrained_kwargs.get("max_memory", None),
        }
        load_kwargs['device_map'] = "auto" # seem there will be some problems using "cuda"
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, **load_kwargs)
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len
        self.conv_mode = "qwen_1_5"
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
        question = prompt
        

        # Check if the video exists
        if os.path.exists(video_path):
            # import pdb;pdb.set_trace()
            video = load_video(video_path, 32)
            video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().to(self.model.device)
            video = [video]

        # try:
        # Run inference on the video and add the output to the list
        qs = question
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self.tokenizer.pad_token_id = 151643
                
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            if "mistral" not in self.model.config._name_or_path.lower():
                output_ids = self.model.generate(
                    inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", 
                    use_cache=True, **generation_kwargs, )
            else:
                output_ids = self.model.generate(
                    inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", 
                    use_cache=True, stopping_criteria=[stopping_criteria], **generation_kwargs)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        

        # import pdb;pdb.set_trace()
        if "mistral" not in self.model.config._name_or_path.lower():
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]

        outputs = outputs.strip()
        return {"text": outputs}
        
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
        # add your custom generation code here
        video_path = decode_and_save_video(params["prompt"]["video"]) # This will save the video to a file and return the path
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        question = prompt
        
        # add streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        

        # Check if the video exists
        if os.path.exists(video_path):
            # import pdb;pdb.set_trace()
            video = load_video(video_path, 32)
            video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().to(self.model.device)
            video = [video]

        # try:
        # Run inference on the video and add the output to the list
        qs = question
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self.tokenizer.pad_token_id = 151643
                
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)


        with torch.inference_mode():
            if "mistral" not in self.model.config._name_or_path.lower():
                all_inputs = {"inputs": input_ids, "images": video, "attention_mask": attention_masks, "modalities": "video",
                    "use_cache": True, **generation_kwargs}
            else:
                all_inputs = {"inputs": input_ids, "images": video, "attention_mask": attention_masks, "modalities": "video",
                    "use_cache": True, "stopping_criteria": [stopping_criteria], **generation_kwargs}
                
            thread = Thread(target=self.model.generate, kwargs={**all_inputs})
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
    model_path = "lmms-lab/LLaVA-NeXT-Video-32B-Qwen"
    device = "cuda"
    model_adapter = LLaVANextVideoQwenAdapter()
    test_adapter(model_adapter, model_path, device, num_gpus=2)
    
"""
# local testing
python -m lmm_engines.huggingface.model.model_llavanextvideoqwen
# connect to wildvision arena
bash start_worker_on_arena.sh lmms-lab/LLaVA-NeXT-Video-32B-Qwen 41411 2
"""