import time
import requests
from ...utils import encode_image, encode_video
from PIL import Image
from io import BytesIO
from .model_adapter import BaseModelAdapter, get_model_adapter, load_adapter
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers.utils import logging
logging.set_verbosity_error() 

def test_adapter(
    model_adapter: BaseModelAdapter,
    model_path: str,
    device: str = "cuda",
    num_gpus: int = 1,
    model_type="image",
):
    matched_adapter = get_model_adapter(model_path)
    print("\n## Testing model match() method...")
    if model_adapter.__class__.__name__ != matched_adapter.__class__.__name__:
        expected_adapter_name = model_adapter.__class__.__name__
        matched_adapter_name = matched_adapter.__class__.__name__
        ERROR_MSG = f"model path is {model_path}, expected model adapter '{expected_adapter_name}', but got model adapter '{matched_adapter_name}', " + \
            f"Please check the `match()` method of '{expected_adapter_name}'."
        raise ValueError(ERROR_MSG)
    print("### Model match() method passed.")
    
    print("\n## Testing model load_model() method...")
    model_adapter = load_adapter(model_path, device=device, num_gpus=num_gpus)
    
    print("\n## load_model() method returned model: ", model_adapter.model)
    print("### Model load_model() method passed.")
    
    if model_type == "image":
        image_url = "https://llava.hliu.cc/file=/nobackup/haotian/tmp/gradio/ca10383cc943e99941ecffdc4d34c51afb2da472/extreme_ironing.jpg"
        image = Image.open(BytesIO(requests.get(image_url).content))
        params = {
            "prompt": {
                "text": "What is unusual about this image?",
                "image": encode_image(image)
            },
            "do_sample": False,
            "top_p": 1.0,
            "max_new_tokens": 200,
        }
    elif model_type == "raw_video":
        video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
        params = {
            "prompt": {
                "text": "What is happening in this video?",
                "video": video_path
            },
            "do_sample": False,
            "top_p": 1.0,
            "max_new_tokens": 200,
        }
    else:
        video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
        encoded_video = encode_video(video_path)
        params = {
            "prompt": {
                "text": "What is happening in this video?",
                "video": encoded_video
            },
            "do_sample": False,
            "top_p": 1.0,
            "max_new_tokens": 200,
        }
    print("\n## Testing model generate() method...")
    generated_text = model_adapter.generate(params)
    print("### Final generated text: \n", generated_text['text'])
    
    print("\n## Testing model generate_stream() method")
    streamer = model_adapter.generate_stream(params)
    generated_text = ""
    for text in streamer:
        added_text = text['text'][len(generated_text):]
        generated_text = text['text']
        time.sleep(0.03)
        print(added_text, end='', flush=True)
    print("\n### Final generated text via stream: \n", generated_text)
    
    