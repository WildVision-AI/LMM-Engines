import torch
from PIL import Image
import os
from .model_adapter import BaseModelAdapter
from ...utils import decode_image, decode_and_save_video
from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline, TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoProcessor
from threading import Thread
from typing import List

from decord import VideoReader
from PIL import Image
from tqdm import tqdm
from typing import List


def load_video(video_file, num_frames=128, cache_dir="cached_video_frames", verbosity="DEBUG"):
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    video_basename = os.path.basename(video_file)
    cache_subdir = os.path.join(cache_dir, f"{video_basename}_{num_frames}")
    os.makedirs(cache_subdir, exist_ok=True)

    cached_frames = []
    missing_frames = []
    frame_indices = []
    
    for i in range(num_frames):
        frame_path = os.path.join(cache_subdir, f"frame_{i}.jpg")
        if os.path.exists(frame_path):
            cached_frames.append(frame_path)
        else:
            missing_frames.append(i)
            frame_indices.append(i) 
            
    vr = VideoReader(video_file)
    duration = len(vr)
    fps = vr.get_avg_fps()
            
    frame_timestamps = [int(duration / num_frames * (i+0.5)) / fps for i in range(num_frames)]
    
    if verbosity == "DEBUG":
        print("Already cached {}/{} frames for video {}, enjoy speed!".format(len(cached_frames), num_frames, video_file))
    # If all frames are cached, load them directly
    if not missing_frames:
        return [Image.open(frame_path).convert("RGB") for frame_path in cached_frames], frame_timestamps

    

    actual_frame_indices = [int(duration / num_frames * (i+0.5)) for i in missing_frames]


    missing_frames_data = vr.get_batch(actual_frame_indices).asnumpy()

    for idx, frame_index in enumerate(tqdm(missing_frames, desc="Caching rest frames")):
        img = Image.fromarray(missing_frames_data[idx]).convert("RGB")
        frame_path = os.path.join(cache_subdir, f"frame_{frame_index}.jpg")
        img.save(frame_path)
        cached_frames.append(frame_path)

    cached_frames.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    return [Image.open(frame_path).convert("RGB") for frame_path in cached_frames], frame_timestamps

def create_image_gallery(images, columns=3, spacing=20, bg_color=(200, 200, 200)):
    """
    Combine multiple images into a single larger image in a grid format.
    
    Parameters:
        image_paths (list of str): List of file paths to the images to display.
        columns (int): Number of columns in the gallery.
        spacing (int): Space (in pixels) between the images in the gallery.
        bg_color (tuple): Background color of the gallery (R, G, B).
    
    Returns:
        PIL.Image: A single combined image.
    """
    # Open all images and get their sizes
    img_width, img_height = images[0].size  # Assuming all images are of the same size

    # Calculate rows needed for the gallery
    rows = (len(images) + columns - 1) // columns

    # Calculate the size of the final gallery image
    gallery_width = columns * img_width + (columns - 1) * spacing
    gallery_height = rows * img_height + (rows - 1) * spacing

    # Create a new image with the calculated size and background color
    gallery_image = Image.new('RGB', (gallery_width, gallery_height), bg_color)

    # Paste each image into the gallery
    for index, img in enumerate(images):
        row = index // columns
        col = index % columns

        x = col * (img_width + spacing)
        y = row * (img_height + spacing)

        gallery_image.paste(img, (x, y))

    return gallery_image


def get_placeholders_for_videos(frames: List, timestamps=[]):
    contents = []
    if not timestamps:
        for i, _ in enumerate(frames):
            contents.append({"text": None, "type": "image"})
        contents.append({"text": "\n", "type": "text"})
    else:
        for i, (_, ts) in enumerate(zip(frames, timestamps)):
            contents.extend(
                [
                    {"text": f"[{int(ts)//60:02d}:{int(ts)%60:02d}]", "type": "text"},
                    {"text": None, "type": "image"},
                    {"text": "\n", "type": "text"}
                ]
            )
    return contents

class AriaAdapter(BaseModelAdapter):
    """The model adapter for Aria"""
    

    def match(self, model_path: str):
        return "aria" in model_path.lower() and "vllm" not in model_path.lower()

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
        from_pretrained_kwargs["torch_dtype"] = torch.bfloat16
        if device == "cuda":
            if "device_map" not in from_pretrained_kwargs:
                from_pretrained_kwargs["device_map"] = "auto"
        self.torch_dtype = from_pretrained_kwargs["torch_dtype"]
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **from_pretrained_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
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
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        if "image" in params["prompt"]:
            image = decode_image(params["prompt"]["image"]) # This image will be decoded into a PIL image
            visual_contents = [{"text": None, "type": "image"}]
            images = [image]
        elif "video" in params["prompt"]:
            video_path = decode_and_save_video(params["prompt"]["video"]) # This will save the video to a file and return the path
            if "frame_num" in generation_kwargs:
                frame_num = generation_kwargs.pop("frame_num")
            else:
                frame_num = 16
            frames, frame_timestamps = load_video(video_path, num_frames=frame_num)
            visual_contents = get_placeholders_for_videos(frames, frame_timestamps)
            images = frames
        
        messages = [
            {
                "role": "user",
                "content": [
                    *visual_contents,
                    {"text": prompt, "type": "text"},
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        do_sample = generation_kwargs.get("do_sample", True)
        if do_sample and generation_kwargs.get("temperature", 1.0) == 0.0:
            do_sample = False
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=self.torch_dtype):
            output = self.model.generate(
                **inputs,
                max_new_tokens=generation_kwargs.get("max_new_tokens", 500),
                stop_strings=["<|im_end|>"],
                tokenizer=self.processor.tokenizer,
                do_sample=do_sample,
                temperature=generation_kwargs.get("temperature", 0.9),
            )
            output_ids = output[0][inputs["input_ids"].shape[1]:]
            result = self.processor.decode(output_ids, skip_special_tokens=True)
        return {"text": result}
        
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
        prompt = params["prompt"]["text"]
        generation_kwargs = params.copy()
        generation_kwargs.pop("prompt")
        if "image" in params["prompt"]:
            image = decode_image(params["prompt"]["image"]) # This image will be decoded into a PIL image
            visual_contents = [{"text": None, "type": "image"}]
            images = [image]
        elif "video" in params["prompt"]:
            video_path = decode_and_save_video(params["prompt"]["video"]) # This will save the video to a file and return the path
            if "frame_num" in generation_kwargs:
                frame_num = generation_kwargs.pop("frame_num")
            else:
                frame_num = 16
            frames, frame_timestamps = load_video(video_path, num_frames=frame_num)
            visual_contents = get_placeholders_for_videos(frames, frame_timestamps)
            images = frames
        
        # add streamer
        streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        
        messages = [
            {
                "role": "user",
                "content": [
                    *visual_contents,
                    {"text": prompt, "type": "text"},
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        do_sample = generation_kwargs.get("do_sample", True)
        if do_sample and generation_kwargs.get("temperature", 1.0) == 0.0:
            do_sample = False
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=self.torch_dtype):
            gen_kwargs = {
                **inputs,
                "max_new_tokens": generation_kwargs.get("max_new_tokens", 500),
                "stop_strings": ["<|im_end|>"],
                "tokenizer": self.processor.tokenizer,
                "do_sample": do_sample,
                "temperature": generation_kwargs.get("temperature", 0.9),
                "streamer": streamer,
            }
            
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            generated_text = ''
            for new_text in streamer:
                generated_text += new_text
                yield {"text": generated_text}
            thread.join()
    
    def get_info(self):
        return {
            "type": "image;video",
            "author": "RhymesAI",
            "organization": "RhymesAI",
            "model_size": "25.3B",
            "model_link": "https://huggingface.co/rhymes-ai/Aria"
        }
    
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "rhymes-ai/Aria"
    device = "cuda"
    model_adapter = AriaAdapter()
    test_adapter(model_adapter, model_path, device, 2)
    
"""
# local testing
python -m lmm_engines.huggingface.model.model_aria
# connect to wildvision arena
bash start_worker_on_arena.sh model_aria 41412 2
"""