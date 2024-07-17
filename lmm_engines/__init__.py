import os
import sys
import json
import random
import atexit
import signal
from functools import partial
from .utils import generation_cache_wrapper, retry_on_failure
from .utils import shorten_messages

ENGINES = ["huggingface", "vllm", "sglang", "openai", "gemini", "mistral", "claude"]
test_image_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test.jpg")
test_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is unusual about this image?",
            },
            {
                "type": "image_url",
                "image_url": test_image_path
            }
        ]
    }
]
    
workers = []
def get_call_worker_func(
    model_name:str, 
    worker_addrs=None, 
    cache_dir=None, 
    use_cache=True,
    num_workers=1,
    num_gpu_per_worker=1,
    dtype="float16",
    engine="huggingface",
    max_retry=5
) -> str:
    """
    Return a function that calls the model worker, takes a list of messages (user, gpt, user, ...) and returns the generated text
    Args:
        model_name: model name
        worker_addrs: worker addresses, if None, launch local workers
        cache_dir: cache directory
        conv_system_msg: conversation system message
        use_cache: use cache or not
        num_workers: number of workers
        num_gpu_per_worker: number of gpus per worker
        dtype: data type
        engine: engine name
    """
    if engine == "openai":
        raise NotImplementedError("OpenAI API is not available yet")
        from .openai_mm import call_worker_openai
        call_model_worker = call_worker_openai
    elif engine == "gemini":
        raise NotImplementedError("Gemini API is not available yet")
        from .gemini import call_worker_gemini
        call_model_worker = call_worker_gemini
    elif engine == "claude":
        raise NotImplementedError("Claude API is not available yet")
        from .claude import call_worker_claude
        call_model_worker = call_worker_claude
    elif engine == "mistral":
        raise NotImplementedError("Mistral API is not available yet")
        from .mistral import call_worker_mistral
        call_model_worker = call_worker_mistral
    elif engine in ["huggingface"]:
        if engine == "huggingface":
            from .huggingface import launch_hf_worker, call_hf_worker
            call_worker_func = call_hf_worker
            launch_worker_func = launch_hf_worker
        elif engine == "vllm":
            raise NotImplementedError("VLLM API is not available yet")
        elif engine == "sglang":
            raise NotImplementedError("SGLang API is not available yet")
        else:
            raise ValueError(f"Internal error: engine {engine} not supported")
        if worker_addrs is None:
            import torch
            
            print(f"Launching model worker {model_name} locally")
            worker_addrs = []
            total_gpus = torch.cuda.device_count()
            if total_gpus < num_workers * num_gpu_per_worker:
                if total_gpus >= num_gpu_per_worker:
                    print(f"Warning: total gpus ({total_gpus}) is less than num_workers * num_gpu_per_worker ({num_workers * num_gpu_per_worker}), using {total_gpus // num_gpu_per_worker} workers instead")
                    num_workers = total_gpus // num_gpu_per_worker
                else:
                    print(f"Error: total gpus ({total_gpus}) is less than num_gpu_per_worker ({num_gpu_per_worker}), exiting...")
                    sys.exit(1)
                    
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                gpus_ids = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
                gpu_ids = [int(gpu_id) for gpu_id in gpus_ids]
            else:
                gpu_ids = list(range(total_gpus))
            start_port = random.randint(31000, 32000)
            for i in range(num_workers):
                worker_addr, worker = launch_worker_func(model_name, 
                    num_gpus=num_gpu_per_worker, 
                    gpu_ids=gpu_ids[i*num_gpu_per_worker:(i+1)*num_gpu_per_worker], 
                    port=start_port+i*10,
                    dtype=dtype)
                worker_addrs.append(worker_addr)
                workers.append(worker)
            atexit.register(lambda: [cleanup_process(proc) for proc in workers])
        else:
            print(f"Using existing worker at {worker_addrs}")
            if not isinstance(worker_addrs, list):
                worker_addrs = [worker_addr]
        call_model_worker = partial(call_worker_func, worker_addrs=worker_addrs)        
    else:
        raise ValueError(f"Engine {engine} not supported, available engines: {ENGINES}")
    
    # wrap the call_model_worker with the model_name and other arguments
    
    call_model_worker = partial(call_model_worker, model_name=model_name)
    
    # test local worker connection
    test_response = call_model_worker(test_messages, temperature=0, max_tokens=1024)
    if not test_response:
        print("Error: failed to connect to the worker, exiting...")
        for worker in workers:
            cleanup_process(worker)
        sys.exit(1)
    else:
        print(f"Successfully connected to the workers")
        print("Test messages: ")
        print(json.dumps(shorten_messages(test_messages), indent=4))
        print("Test response: \n", test_response)
        
    # add cache wrapper
    if use_cache:
        call_model_worker = generation_cache_wrapper(call_model_worker, model_name, cache_dir)
    else:
        print("Cache is disabled")
    
    call_model_worker = retry_on_failure(call_model_worker, num_retries=max_retry)
    return call_model_worker

def cleanup_process(proc):
    # os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    os.kill(proc.pid, signal.SIGTERM)
    print("Subprocess terminated.")
    
def cleanup_all_workers():
    for worker in workers:
        cleanup_process(worker)
    print("All workers terminated.")