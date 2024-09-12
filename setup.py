import json
from setuptools import setup, find_packages

model_specific_requirement_files = {
    "cogvlm2-video": "lmm_engines/huggingface/model/videollm_utils/cogvlm/requirements.txt",
    "qwen2vl": "lmm_engines/model_requirements/qwen2vl.txt",
    "pixtral": "lmm_engines/model_requirements/pixtral.txt",
}

extra_requirements = {}
for model, requirement_file in model_specific_requirement_files.items():
    with open(requirement_file) as f:
        extra_requirements[model] = f.read().splitlines()
    
setup(
    name='lmm-engines',
    version='0.0.1',
    description='',
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    packages=find_packages(),
    url='https://github.com/jdf-prog/LMM-Engines',
    install_requires=[
        "fire",
        "openai",
        "google-generativeai",
        "accelerate",
        "transformers",
        "torch",
        "Pillow",
        "torch",
        "tqdm",
        "numpy",
        "requests",
        "sentencepiece",
        "icecream",
        "prettytable",
        "mistralai",
        "anthropic",
        "peft>=0.11.0",
        "decord",
        "matplotlib",
        "transformers_stream_generator",
        "einops",
        "timm",
        "av",
        "opencv-python",
        'uvicorn',
        'datasets',
        'fastapi',
    ],
    extras_require={
        **extra_requirements,
        "vllm": ["vllm"],
        "sglang": ["sglang[all]"],
    }
)
