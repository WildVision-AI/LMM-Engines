## Contributing a new model

### Step 1: Understand the model adapter structure
Each model's custom inference logic should all be implemented in the `model_{model_name}.py` file, where you need to inherit the `BaseModelAdapter` class and implement the following methods:
- `match()` method: to match whether a custom model path is compatible with the model adapter.
- `load_model()` method: to load the model from the custom model path.
- `generate()` method: to generate the response based on the input messages. (no streaming)
- `generate_streaming()` method: to generate the response based on the input messages. (streaming)

### Step 2: Implement the model adapter
Example implementation [`model_tinyllava.py`](./model_tinyllava.py) is a template for contributing a new model to the `lmm_engines` package. 
you should refer to this example and implement your own model adapter.

**Important notes:**
- Due to the historical reasons, there are some existing load_model implementations in [`model_adapter_old.py`](./model_adapter_old.py), named `load_{model_name}_pretrained_model`. You can refer to this method for easier implementation.
- Due to the historical reasons, there are some existing generate implementations in `{model_name}_adapter.py`, called `generate_stream_{model_name}`. Although they are called `generate_stream`, they are actually used for non-streaming generation. You can refer to this method for the `generate` method implementation in the adapter.
- Please refer to [`model_tinyllava.py`](./model_tinyllava.py) for how to implement the `generate_stream` method. It's totally compatible with the huggingface generation pipeline, so that implementation is universally compatible with all models.
- You can implement the chat template format in the `generate` or `generate_stream` method, instead of using the previous `get_default_conv_template` method. That's kind of legacy code, and we are trying to remove it.
- **It's better to simply include the naive implementation from the model's hugging face model page, without introducing redundant codes**

### Step 3: Test the model adapter
After implementing the model adapter, you also need to add the model testing logic at the bottom of the file, `model_{model_name}.py`, like this:
```python
if __name__ == "__main__":
    from .unit_test import test_adapter
    from PIL import Image
    model_path = "bczhou/tiny-llava-v1-hf"
    device = "cuda:0"
    model_adapter = TinyLLaVAAdapter()
    model_adapter.load_model(model_path, device, from_pretrained_kwargs)
    test_adapter(model_adapter, model_path, device), device), device), device), device), device))
```
Then you can test the model adapter by running the following command:
```bash
python -m lmm_engines.huggingface.model.model_{model_name}
```

### Step 4: Register the model adapter
Finally, you need to register the model adapter at the bottom of the [`model_adapter.py`](./model_adapter.py) file, like this:
```python
from .model_tinyllava import TinyLLaVAAdapter
register_model_adapter(TinyLLaVAAdapter)
```
(Note: Some todo models are already registered in the `model_adapter.py` file, so you don't need to register them again.)

### Step 5: test the whole pipeline
After implementing the model adapter, you need to test the whole pipeline by running the following command:
```python
from lmm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="{your_model_name}",
    engine="huggingface",
    num_workers=1,
    num_gpu_per_worker=1,
    dtype="float16",
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

### Step 6: Submit a pull request
You need to make sure that the model adapter is working correctly before submitting a pull request.