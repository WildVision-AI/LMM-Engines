import subprocess
import threading
import time
import os
import signal
import os
import json
import base64
import hashlib
import requests
import traceback
import threading
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Union, List
from typing import List
from transformers import AutoTokenizer
class SubprocessMonitor:
    def _monitor(self):
        while True:
            if self.proc.poll() is not None:
                print("Subprocess has exited with code", self.proc.returncode)
                os.kill(os.getpid(), signal.SIGTERM)  # Exit the main process
                break
            time.sleep(5)
            
    def __init__(self, command, **kwargs):
        print("Launching subprocess with command:\n", " ".join(command))
        self.proc = subprocess.Popen(command, **kwargs)
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
    
    def __getattr__(self, name):
        return getattr(self.proc, name)
    
class ChatTokenizer:
    def __init__(self, model_name):
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.system_message = None
        if self.tokenizer.chat_template:
            self.apply_chat_template = self.apply_chat_template_default
            print("Using hugging face chat template for model", model_name)
            self.chat_template_source = "huggingface"
        else:
            raise NotImplementedError("Chat template not implemented for model", model_name)
        print("Example prompt: \n", self.example_prompt())
        
    def apply_chat_template_default(
        self, 
        messages:List[str],
        add_generation_prompt:bool=True,
        chat_template:str=None
    ):
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            chat_template=chat_template,
        )
        return prompt
    
    def example_prompt(self):
        example_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good, how about you?"},
        ]
        return self.apply_chat_template(example_messages)
    
    def __call__(self, messages:List[str], **kwargs):
        return self.apply_chat_template(messages, **kwargs)


def encode_image(image:Image.Image) -> str:
    im_file = BytesIO()
    image.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()
    im_64 = base64.b64encode(im_bytes).decode("utf-8")
    return json.dumps(im_64)

def convert_messages(messages:List[dict]) -> List[dict]:
    new_messages = []
    for message in messages:
        new_message = {
            "role": message["role"],
            "content": []
        }
        for _content in message["content"]:
            if _content["type"] == "image_url":
                if _content["image_url"].startswith("http"):
                    response = requests.get(_content["image_url"])
                    encoded_image = encode_image(Image.open(BytesIO(response.content)))
                else:
                    with open(_content["image_url"], "rb") as f:
                        encoded_image = encode_image(Image.open(f))
                new_message["content"].append({
                    "type": "image",
                    "image": encoded_image
                })
            elif _content["type"] == "image":
                if isinstance(_content["image"], Image.Image):
                    encoded_image = encode_image(_content["image"])
                new_message["content"].append({
                    "type": "image",
                    "image": encoded_image
                })
            elif _content["type"] == "video_url":
                response = requests.get(_content["video_url"])
                encoded_video = base64.b64encode(response.content).decode("utf-8")
                new_message["content"].append({
                    "type": "video",
                    "video": encoded_video
                })
            elif _content["type"] == "video":
                if isinstance(_content["video"], bytes):
                    encoded_video = base64.b64encode(_content["video"]).decode("utf-8")
                new_message["content"].append({
                    "type": "video",
                    "video": encoded_video
                })
            else:
                new_message["content"].append(_content)
        new_messages.append(new_message)
    return new_messages

def shorten_messages(messages:List[dict]):
    new_messages = []
    for message in messages:
        new_message = {
            "role": message["role"],
            "content": []
        }
        for _content in message["content"]:
            if _content["type"] != 'text' and len(str(_content[_content["type"]])) > 100:
                new_message["content"].append({
                    "type": _content["type"],
                    _content["type"]: str(_content[_content["type"]])[:100] + "..."
                })
            else:
                new_message["content"].append(_content)
        new_messages.append(new_message)
    return new_messages


cache_dict = None
def generation_cache_wrapper(call_model_worker, model_name, cache_dir=None):
    print(f"Using cache for model {model_name}")
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{model_name}.jsonl"
    else:
        # cache_file = Path(os.path.abspath(__file__)).parent / "generation_cache" / f"{model_name}.jsonl"
        cache_file = Path(os.path.expanduser(f"~/lmm_engines/generation_cache/{model_name}.jsonl"))
    if cache_file.exists():
        print(f"Cache file exists at {cache_file}")
    print(f"Each single input will be cached in hash-input:output format in {cache_file}")
    def wrapper(messages:List[dict], **generate_kwargs):
        global cache_dict
        if cache_dir is not None:
            cache_file = Path(cache_dir) / f"{model_name}.jsonl"
        else:
            # cache_file = Path(os.path.abspath(__file__)).parent.parent / "generation_cache" / f"{model_name}.jsonl"
            cache_file = Path(os.path.expanduser(f"~/lmm_engines/generation_cache/{model_name}.jsonl"))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_dict is None:
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    cache_dict = [json.loads(line) for line in f.readlines()]
                cache_dict = {list(item.keys())[0]: list(item.values())[0] for item in cache_dict}
            else:
                cache_dict = {}
        shorted_messages = shorten_messages(messages)
        shorted_messages_json_str = json.dumps(shorted_messages, ensure_ascii=False)
        messages_hash = hashlib.md5(shorted_messages_json_str.encode()).hexdigest()
        if messages_hash in cache_dict:
            return cache_dict[messages_hash]["output"]
        else:
            generated_text = call_model_worker(messages, **generate_kwargs)
            cache_dict[messages_hash] = {"input": shorted_messages, "output": generated_text, "model_name": model_name, 'tstamp': time.time(), "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "generate_kwargs": generate_kwargs}
            with open(cache_file, "a+") as f:
                f.write(json.dumps({messages_hash: cache_dict[messages_hash]}, ensure_ascii=False) + "\n")
            return generated_text
    return wrapper

class MaxRetriesExceededError(Exception):
    pass

def retry_on_failure(call_model_worker, num_retries=5):
    def wrapper(*args, **kwargs):
        for i in range(num_retries):
            try:
                return call_model_worker(*args, **kwargs)
            except Exception as e:
                print("Error in call_model_worker, retrying... (Error: {})".format(e))
                time.sleep(1)
                if i == num_retries - 1 and not isinstance(e, TimeoutError):
                    # format dump of the last error and
                    print(traceback.format_exc())
        raise MaxRetriesExceededError("Max retries exceeded for call_model_worker")
    return wrapper

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

def with_timeout(timeout):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [TimeoutError(f"Function call timed out (timeout={timeout})")]
            stop_event = threading.Event()

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                stop_event.set()
                raise TimeoutError(f"Function call timed out (timeout={timeout})")
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator