# LMM-Engines

## Installation
- Install the requirements
```bash
pip install -e .
pip install flash-attn --no-build-isolation # optional, for faster inference
```
- for connecting to Wildvision Arena, you need to install bore
```bash
bash install_bore.sh
```
- some models may require additional dependencies, see the top of [`setup.py`](./setup.py) for details. To install extra dependencies for a specific model, you can run
```bash
pip install -e .[cogvlm2-video] # for cogvlm2-video
```
(Note: the extra dependencies for different models might conflict with each other, so you should better create a new virtual environment for each model.)

## Usage

### Local testing
```bash
python -m lmm_engines.huggingface.model.dummy_image_model
python -m lmm_engines.huggingface.model.dummy_video_model
# python -m lmm_engines.huggingface.model.model_tinyllava # example
```

### Connect to Wildvision Arena and be one arena competitor

First run `bash install_bore.sh` once to install bore.
```bash
bash start_worker_on_arena.sh ${model_name} ${model_port} ${num_gpu}
# Example
bash start_worker_on_arena.sh dummy_image_model 41411 1
```
Then your worker shall be registered to the arena. You can check it by visiting 🤗 [WildVision/vision-arena](https://huggingface.co/spaces/WildVision/vision-arena)

See `## Controbute a model` section for how to contribute your own model.


### Start a new worker for local inference
```bash
CUDA_VISIBLE_DEVICES=0 python -m lmm_engines.huggingface.model_worker --model-path dummy_image_model --port 31004 --worker http://127.0.0.1:31004 --host=127.0.0.1 --no-register
```
Then call the worker
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

Or you can start a new worker automatically, fusing the above two steps all in one. model worker will close automatically after the python script ends.
```python
from lmm_engines import get_call_worker_func
# start a new worker
call_worker_func = get_call_worker_func(
    model_name="dummy_image_model", # 
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
print(call_worker_func(test_messages, **generation_kwargs))
```

- output cache
set `use_cache=True` to enable output cache. The cache will be stored in `~/lmm_engines/generation_cache/{model_name}.jsonl` by default.


## Controbute a model

- If you are contributing a new image model, copy the [lmm_engines/huggingface/model/dummy_image_model.py](./lmm_engines/huggingface/model/dummy_image_model.py) and modify it. (the `type` in `get_info` should be `image`)
- If you are contributing a new video model, copy the [lmm_engines/huggingface/model/dummy_video_model.py](./lmm_engines/huggingface/model/dummy_video_model.py) and modify it. (the `type` in `get_info` should be `video`)
- If the model is both an image and video model, set the `type` in `get_info` to `image;video`. (example implementation: [lmm_engines/huggingface/model/model_qwen2vl.py](./lmm_engines/huggingface/model/model_qwen2vl.py))
- Four functions to implement:
    - `load_model(self, model_path: str, device: str, from_pretrained_kwargs: Dict[str, Any]) -> None`
    - `generate(self, messages: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]`
    - `generate_image(self, image: Image.Image, **kwargs) -> Image.Image`
    - `generate_video(self, video: List[Image.Image], **kwargs) -> List[Image.Image]`
    - `get_info(self) -> Dict[str, Any]`
- test the model adapter: see [lmm_engines/huggingface/README.md](./lmm_engines/huggingface/README.md)
- add registration at the bottom of [`lmm_engines/huggingface/model/model_adapter.py`](./lmm_engines/huggingface/model/model_adapter.py)
- add model-specific dependencies in [`./lmm_engines/model_requirements/{model_name}.txt`](./lmm_engines/model_requirements/)
- Connect to Wildvision Arena and be one arena competitor: `bash start_worker_on_arena.sh ${model_name} ${model_port}`

(Note: we don't care the internal details of these 5 functions, as long as it can receive params and return the expected results as specified in the function signature.)


More details to see [lmm_engines/huggingface/README.md](./lmm_engines/huggingface/README.md)

## TODO
### Transfering models from old arena codes into lmm-engines
- [x] add support for [model_tinyllava.py](./lmm_engines/huggingface/model/model_tinyllava.py) (Example implementation by dongfu)
- [x] add support for [model_llavanextvideo.py](./lmm_engines/huggingface/model/model_llavanextvideo.py) (Example implementation by dongfu)
- [x] add support for [model_llavanextvideoqwen.py](./lmm_engines/huggingface/model/model_llavanextvideoqwen.py) (ontributed by dongfu, on 2024-07-28)
- [x] add support for [model_nvidia_api.py](./lmm_engines/huggingface/model/model_nvidia_api.py) (contributed by jing gu, on 2024-07-28)
- [x] add support for [model_internvl2.py](./lmm_engines/huggingface/model/model_internvl2.py) (contributed by chenhui, on 2024-08-05)
- [x] add support for [model_qwen2vl.py](./lmm_engines/huggingface/model/model_qwen2vl.py) (contributed by [Yingzi Ma](https://github.com/gray311), on 2024-08-05)
- [ ] add support for [model_bunny.py](./lmm_engines/huggingface/model/model_bunny)
- [ ] add support for [model_deepseekvl.py](./lmm_engines/huggingface/model/model_deepseekvl)
- [ ] add support for [model_idefics.py](./lmm_engines/huggingface/model/model_idefics)
- [ ] add support for [model_instructblip.py](./lmm_engines/huggingface/model/model_instructblip)
- [ ] add support for [model_lita.py](./lmm_engines/huggingface/model/model_lita)
- [ ] add support for [model_llavanext.py](./lmm_engines/huggingface/model/model_llavanext)
- [ ] add support for [model_llava.py](./lmm_engines/huggingface/model/model_llava)
- [ ] add support for [model_qwenvl.py](./lmm_engines/huggingface/model/model_qwenvl)
- [ ] add support for [model_uform.py](./lmm_engines/huggingface/model/model_uform)
- [ ] add support for [model_videollama2.py](./lmm_engines/huggingface/model/model_videollama2)
- [x] add support for [model_videollava.py](./lmm_engines/huggingface/model/model_videollava) (contributed by dongfu, on 2024-07-28)
- [ ] add support for [model_yivlplus.py](./lmm_engines/huggingface/model/model_yivlplus)
- [ ] add support for [model_yivl.py](./lmm_engines/huggingface/model/model_yivl)
- [ ] add support for [model_reka.py](./lmm_engines/huggingface/model/model_reka)
- [ ] add support for [model_llava_v1_5.py](./lmm_engines/huggingface/model/model_llava_v1_5)
- [ ] add support for [model_llava_v1_6.py](./lmm_engines/huggingface/model/model_llava_v1_6)
- [ ] add support for [model_minicpm.py](./lmm_engines/huggingface/model/model_minicpm)
- [ ] add support for [model_minicpmapi.py](./lmm_engines/huggingface/model/model_minicpmapi)
- [ ] add support for [model_llavaapi.py](./lmm_engines/huggingface/model/model_llavaapi)
- [ ] add support for [model_cogvlm.py](./lmm_engines/huggingface/model/model_cogvlm)
- [ ] add support for [model_qwenvlapi.py](./lmm_engines/huggingface/model/model_qwenvlapi)
- [ ] add support for [model_openai.py](./lmm_engines/huggingface/model/model_openai)
- [ ] add support for [model_claude.py](./lmm_engines/huggingface/model/model_claude)
- [ ] add support for [model_gemini.py](./lmm_engines/huggingface/model/model_gemini)
