# LMM-Engines

CUDA_VISIBLE_DEVICES=0 python3 -m lmm_engines.huggingface.model_worker --model-path Qwen/Qwen-VL-Chat --controller http://127.0.0.1:21002 --port 31004 --worker http://127.0.0.1:31004 --host=127.0.0.1 &


## Usage

- Start a new worker for local inference
```bash
python -m lmm_engines.huggingface.model_worker --model-path bczhou/tiny-llava-v1-hf --port 31004 --worker http://127.0.0.1:31004 --host=127.0.0.1 --no-register
```

- Start a new worker for connecting to a controller
```bash
python -m lmm_engines.huggingface.model_worker --model-path bczhou/tiny-llava-v1-hf --controller {controller_address} --port 31004 --worker http://127.0.0.1:31004 --host=127.0.0.1
```


- call the worker
```python
from lmm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    worker_addrs=["http://127.0.0.1:31004"],
    use_cache=False
)
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
                "image_url": "https://llava.hliu.cc/file=/nobackup/haotian/tmp/gradio/ca10383cc943e99941ecffdc4d34c51afb2da472/extreme_ironing.jpg"
            }
        ]
    }
]
generation_kwargs = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 200,
}
call_worker_func(test_messages, **generation_kwargs)
```

- all in one for custom inference in your python script
```python
from lmm_engines import get_call_worker_func
# start a new worker
call_worker_func = get_call_worker_func(
    model_name="bczhou/tiny-llava-v1-hf",
    engine="huggingface",
    num_workers=1,
    num_gpu_per_worker=1,
    dtype="float16",
    use_cache=False
)
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
                "image_url": "https://llava.hliu.cc/file=/nobackup/haotian/tmp/gradio/ca10383cc943e99941ecffdc4d34c51afb2da472/extreme_ironing.jpg"
            }
        ]
    }
]
generation_kwargs = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 200,
}
# call the worker
call_worker_func(test_messages, **generation_kwargs)
```

- output cache
set `use_cache=True` to enable output cache. The cache will be stored in `~/lmm_engines/generation_cache/{model_name}.jsonl` by default.