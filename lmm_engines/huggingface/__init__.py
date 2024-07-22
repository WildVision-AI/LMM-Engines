import os
import time
import torch
import requests
import json
import random
from pathlib import Path
from typing import List
from ..utils import SubprocessMonitor, convert_messages, with_timeout

worker_initiated = False

def launch_hf_worker(
    model_name: str,
    num_gpus: int=None,
    gpu_ids: List[int]=None,
    dtype: str="float16",
    port: int=34200,
    host: str="127.0.0.1",
) -> str:
    """
    Launch a model worker and return the address
    Args:
        model_name: the model name to launch
    Returns:
        the address of the launched model
    """
    print(f"Launching model {model_name}")
    worker_addr = f"http://{host}:{port}"
    log_file = Path(os.path.abspath(__file__)).parent / "logs" / f"{model_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if gpu_ids:
        num_gpus = len(gpu_ids)
    else:
        if not num_gpus:
            num_gpus = torch.cuda.device_count()
            print(f"Warning: num_gpus or gpu_ids not provided, using {num_gpus} GPUs")
        gpu_ids = list(range(num_gpus))
        
    env = os.environ.copy()
    # Set the CUDA_VISIBLE_DEVICES environment variable
    env["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu_id) for gpu_id in gpu_ids])
    print(num_gpus, gpu_ids)
    
    
    proc = SubprocessMonitor([
        "python3", "-m", "lmm_engines.huggingface.model_worker",
        "--model-path", model_name,
        "--dtype", dtype,
        "--port", str(port),
        "--host", host,
        "--worker", worker_addr,
        "--gpus", ",".join([str(gpu_id) for gpu_id in gpu_ids]),
        "--no-register"
    ], env=env)
    print(f"Launched HuggingFace model {model_name} at address {worker_addr}")
    return f"http://127.0.0.1:{port}", proc

    
def call_hf_worker(messages:List[dict], model_name:str, worker_addrs:List[str], timeout:int=60, **generate_kwargs) -> str:
    """
    Call a model worker with a list of messages
    Args:
        messages: a list of messages
            [
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": "Hello, how are you?"
                    },
                    {
                        "type": "image",
                        "image": "{base64 encoded image}"
                    },
                    {
                        "type": "image_url",
                        "image_url": "https://example.com/image.jpg"
                    },
                    {
                        "type": "video",
                        "video": "{base64 encoded video}"
                    },
                    {
                        "type": "video_url",
                        "video_url": "https://example.com/video.mp4"
                    },
                    ...
                ]
            ]
        model_name: the model name to call
        worker_addrs: a list of worker addresses
        conv_system_msg: a system message to prepend to the conversation
        generate_kwargs: additional keyword arguments for the generation
    """
    global worker_initiated
    assert isinstance(messages, list), "Messages must be a list"
    assert isinstance(messages[0], dict), "Messages must be a list of dictionaries"
    messages = convert_messages(messages)
    
    num_images = sum([1 for message in messages[0]["content"] if message["type"] == "image"])
    num_texts = sum([1 for message in messages[0]["content"] if message["type"] == "text"])
    num_videos = sum([1 for message in messages[0]["content"] if message["type"] == "video"])
    
    # for now, we assert:
    assert len(messages) == 1, "Only one message is supported for now"
    assert num_images <= 1, "Only one image is supported for now"
    assert num_texts <= 1, "Only one text is supported for now"
    assert num_texts >= 1, "At least one text is required"
    assert num_videos <= 1, "Only one video is supported for now"
    assert num_images + num_videos <= 1, "Only one image or video is supported for now"
    
    # huggingface do not accept max_tokens, only max_new_tokens
    if "max_tokens" in generate_kwargs:
        if "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = generate_kwargs["max_tokens"]
        del generate_kwargs["max_tokens"]
        
    text = [_content["text"] for _content in messages[0]["content"] if _content["type"] == "text"][0]
    params = {
        "prompt": {
            "text": text,
        },
        **generate_kwargs
    }
    if num_images == 1:
        image = [_content["image"] for _content in messages[0]["content"] if _content["type"] == "image"][0]
        params["prompt"]["image"] = image
    elif num_videos == 1:
        video = [_content["video"] for _content in messages[0]["content"] if _content["type"] == "video"][0]
        params["prompt"]["video"] = video
    else:
        raise ValueError("No image or video provided")
    worker_addr = random.choice(worker_addrs)

    @with_timeout(timeout)
    def get_response():
        global worker_initiated
        while True:
            try:
                worker_details = requests.post(worker_addr + "/model_details").json()
                if model_name not in worker_details["model_names"] and model_name.split('/')[-1] not in worker_details["model_names"]:
                    raise ValueError(f"Model {model_name} not found in worker {worker_addr}. Available models on this address: {worker_details['model_names']}")
                # starlette StreamingResponse
                response = requests.post(
                    worker_addr + "/worker_generate",
                    json=params,
                    stream=True,
                    # timeout=timeout,
                )
                if response.status_code == 200:
                    worker_initiated = True
                break
            except requests.exceptions.ConnectionError as e:
                if not worker_initiated:
                    print("Worker not initiated, waiting for 5 seconds...")
                else:                
                    print("Connection error, retrying...")
                time.sleep(5)
            except requests.exceptions.ReadTimeout as e:
                print("Read timeout, adding 10 seconds to timeout and retrying...")
                timeout += 10
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                print("Unknown request exception: ", e, "retrying...")
                time.sleep(5)
        try:
            generated_text = json.loads(response.content.decode("utf-8"))['text']
            generated_text = generated_text.strip("\n ")
        except Exception as e:
            generated_text = response.content.decode("utf-8")
            # print("Error in worker response: ", e)
            # generated_text = "**RESPONSE DECODING ERROR**"
        return generated_text
    
    return get_response()
