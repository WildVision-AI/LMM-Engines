"""
Global constants.
"""

from enum import IntEnum
import os
from datetime import timedelta

REPO_PATH = os.path.dirname(os.path.dirname(__file__))
CONVERSATION_SAVE_DIR = 'conversation_data'

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)
##### For the gradio web server
SERVER_ERROR_MSG = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)

MODERATION_MSG = "$MODERATION$ YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES."
CONVERSATION_LIMIT_MSG = "YOU HAVE REACHED THE CONVERSATION LENGTH LIMIT. PLEASE CLEAR HISTORY AND START A NEW CONVERSATION."
INACTIVE_MSG = "THIS SESSION HAS BEEN INACTIVE FOR TOO LONG. PLEASE REFRESH THIS PAGE."
SLOW_MODEL_MSG = "⚠️  Both models will show the responses all at once. Please stay patient as it may take over 30 seconds."
RATE_LIMIT_MSG = "**RATE LIMIT PER DAY OF THIS MODEL IS REACHED. PLEASE COME BACK LATER OR TRY OTHER MODELS.**"
# Maximum input length
INPUT_CHAR_LEN_LIMIT = int(os.getenv("FASTCHAT_INPUT_CHAR_LEN_LIMIT", 12000))
# Maximum conversation turns
CONVERSATION_TURN_LIMIT = 50
# Session expiration time
SESSION_EXPIRATION_TIME = 3600
# Rate Limite Per Day
MAX_REQUESTS_PER_DAY = 100
RATE_LIMIT_PERIOD = timedelta(days=1)
# The output dir of log files
# LOGDIR = os.getenv("WILDVISION_ARENA_LOGDIR", "/home/yuchenl/vision-arena-logs/")
LOGDIR = os.getenv("WILDVISION_ARENA_LOGDIR", "../log")
# The WEB_IMG_FOLDER
WEB_IMG_FOLDER = os.getenv("WEB_IMG_FOLDER", "/home/yuchenl/http_img/")
# The WEB_IMG_URL_ROOT
WEB_IMG_URL_ROOT = os.getenv("WEB_IMG_URL_ROOT", "http://34.19.37.54:5090")
# CPU Instruction Set Architecture
CPU_ISA = os.getenv("CPU_ISA")

HEADER_MD = """
# ⚔️  WildVision Arena ⚔️ : Benchmarking Multimodal LLMs in the Wild
by WildVision Team @ AI2 + UCSB + UWaterloo
"""

INFO_MD = """
## 📈 Leaderboard (Elo Rating) is live now! 

## 👀 Supported Models

**GPT-4V, Gemini Pro Vision, Llava (v1.6-7/34b), CogVLM-Chat, Qwen-VL-Chat, MiniCPM-V, InstructBLIP, and more!**

<details>
<summary>Show details</summary>

- GPT-4V: https://openai.com/research/gpt-4v-system-card
- Gemini Pro Vision: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini#gemini-pro-vision  
- Claude 3: https://www.anthropic.com/claude
- Llava-1.6-vicuna-7b: https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b
- Llava-1.6-34b: https://huggingface.co/liuhaotian/llava-v1.6-34b
- CogVLM-Chat: https://huggingface.co/THUDM/cogvlm-chat-hf  
- Qwen-VL-Chat: https://huggingface.co/Qwen/Qwen-VL-Chat
- MiniCPM-V: https://huggingface.co/openbmb/MiniCPM-V
- InstructBLIP-vicuna-7b: https://huggingface.co/Salesforce/instructblip-vicuna-7b
- Uform-Gen2-Qwen-500M: https://huggingface.co/uform/uform-gen2-qwen-500m
- Tiny-Llava-v1-HF: https://huggingface.co/bczhou/tiny-llava-v1-hf
- + More! 

</details>
"""

ABOUT_US_MD = """
# About Us

This is a research project by the WildVision team at Allen Institute for AI (AI2), UCSB, and UWaterloo. 

# WildVision Team  

[Yujie Lu](https://yujielu10.github.io/), [Dongfu Jiang](https://jdf-prog.github.io/), [Wenhu Chen](https://wenhuchen.github.io/), [William Wang](https://sites.cs.ucsb.edu/~william/), [Yejin Choi](https://homes.cs.washington.edu/~yejin/), and [Bill Yuchen Lin](https://yuchenlin.xyz)

# Contact: 

Email: yuchenl@allenai.org (Yuchen), yujielu@umail.ucsb.edu (Yujie)

# Sponsorship

We are looking for sponsorship to support this project for the long term. Please contact us if you are interested in supporting this project.
"""

##### For the controller and workers (could be overwritten through ENV variables.)
SAMPLING_WEIGHTS = {
    "gpt-4-vision-preview": 50,
    "gpt-4o": 100,
    "gemini-pro-vision": 10,
    "gemini-1.5-flash-latest": 100,
    "gemini-1.5-pro-latest": 100,
    "cogvlm-chat-hf": 1,
    "llava-v1.6-vicuna-7b": 2,
    "llava-v1.6-vicuna-13b": 0,
    "llava-v1.6-34b": 50,
    # "MiniCPM-V": 2,
    "MiniCPM-V-2": 2,
    "minicpm-llama3-v": 50,
    "Qwen-VL-Chat": 2,
    "instructblip-vicuna-7b": 0,
    "uform-gen2-qwen-500m": 1,
    "tiny-llava-v1-hf": 4,
    "claude-3-opus-20240229": 50,
    "claude-3-sonnet-20240229": 50,
    "claude-3-5-sonnet-20240620": 100,
    "claude-3-haiku-20240307": 50,
    "deepseek-vl-7b-chat": 10,
    "Reka-Flash": 50,
    "Reka-Core": 300,
    "idefics2-8b-chatty": 1,
    "Bunny-v1_0-3B": 5,
    "yi-vl-plus": 100,
    "qwen-vl-max": 100,
    "qwen-vl-plus": 100,
}
# each model can be a target of the other models.
BATTLE_TARGETS = {k: set(SAMPLING_WEIGHTS.keys()) - {k} for k in SAMPLING_WEIGHTS}
# BATTLE_TARGETS = {}
# models = SAMPLING_WEIGHTS.keys() - {"minicpm-v", "qwen-vl-chat", "cogvlm-chat-hf"}
# for k in models:
#     if k == "llava-v1.6-vicuna-7b":
#         BATTLE_TARGETS[k] = models - {k}
#     else:
#         BATTLE_TARGETS[k] = {"llava-v1.6-vicuna-7b"} 

# print(BATTLE_TARGETS)
# """
# >>> print(BATTLE_TARGETS)
# {'llava-v1.6-vicuna-13b': {'llava-v1.6-vicuna-7b'}, 'gemini-pro-vision': {'llava-v1.6-vicuna-7b'}, 'llava-v1.6-vicuna-7b': {'gemini-pro-vision', 'llava-v1.6-vicuna-13b', 'llava-v1.6-34b', 'gpt-4-vision-preview'}, 'llava-v1.6-34b': {'llava-v1.6-vicuna-7b'}, 'gpt-4-vision-preview': {'llava-v1.6-vicuna-7b'}}
# """
# BATTLE_TARGETS = {
#     "gpt-4-vision-preview": {"gemini-pro-vision", "llava-v1.6-34b", "cogvlm-chat-hf", "qwen-vl-chat"},
#     "gemini-pro-vision": {"gpt-4-vision-preview", "llava-v1.6-34b", "cogvlm-chat-hf", "qwen-vl-chat"},
#     # "llava-v1.5-13b": {"gpt-4-vision-preview", "gemini-pro-vision", "llava-v1.6-34b", "cogvlm-chat-hf", "qwen-vl-chat"},
#     # "llava-v1.5-7b": {"gpt-4-vision-preview", "gemini-pro-vision", "llava-v1.6-34b"},
#     "llava-v1.6-34b": {"gpt-4-vision-preview", "gemini-pro-vision", "cogvlm-chat-hf", "qwen-vl-chat"},
#     "cogvlm-chat-hf": {"gpt-4-vision-preview", "gemini-pro-vision", "llava-v1.6-34b", "qwen-vl-chat"},
#     "qwen-vl-chat": {"gpt-4-vision-preview", "gemini-pro-vision", "llava-v1.6-34b"},
# }
SAMPLING_BOOST_MODELS = []
# SAMPLING_BOOST_MODELS = ["cogvlm-chat-hf", "MiniCPM-V", "Qwen-VL-Chat"]

# outage models won't be sampled.
OUTAGE_MODELS = ["llava-v1.6-vicuna-13b"]
# OUTAGE_MODELS = ["llava-v1.6-vicuna-13b"]
# SAMPLING_BOOST_MODELS = ["cogvlm-chat-hf", "minicpm-v", "qwen-vl-chat"]

# SAMPLING_WEIGHTS = {
#     # tier 0
#     "gpt-4": 4,
#     "gpt-4-0314": 4,
#     "gpt-4-turbo": 4,
#     "gpt-3.5-turbo-0613": 2,
#     "gpt-3.5-turbo-1106": 2,
#     "claude-2.1": 4,
#     "claude-2.0": 2,
#     "claude-1": 2,
#     "claude-instant-1": 4,
#     "gemini-pro": 4,
#     "pplx-7b-online": 4,
#     "pplx-70b-online": 4,
#     "solar-10.7b-instruct-v1.0": 2,
#     "mixtral-8x7b-instruct-v0.1": 4,
#     "openhermes-2.5-mistral-7b": 2,
#     "dolphin-2.2.1-mistral-7b": 2,
#     "wizardlm-70b": 2,
#     "starling-lm-7b-alpha": 2,
#     "tulu-2-dpo-70b": 2,
#     "yi-34b-chat": 2,
#     "zephyr-7b-beta": 2,
#     "openchat-3.5": 2,
#     "chatglm3-6b": 2,
#     # tier 1
#     "deluxe-chat-v1.2": 2,
#     "llama-2-70b-chat": 1.5,
#     "llama-2-13b-chat": 1.5,
#     "codellama-34b-instruct": 1.5,
#     "vicuna-33b": 4,
#     "vicuna-13b": 1.5,
#     "wizardlm-13b": 1.5,
#     "qwen-14b-chat": 1.5,
#     "mistral-7b-instruct": 1.5,
#     # tier 2
#     "vicuna-7b": 1.0,
#     "llama-2-7b-chat": 1.0,
#     "chatglm2-6b": 1.0,
#     # deprecated
#     "zephyr-7b-alpha": 1.5,
#     "codellama-13b-instruct": 1.0,
#     "mpt-30b-chat": 1.5,
#     "guanaco-33b": 1.0,
#     "fastchat-t5-3b": 0.5,
#     "alpaca-13b": 0.5,
#     "mpt-7b-chat": 0.1,
#     "oasst-pythia-12b": 0.1,
#     "RWKV-4-Raven-14B": 0.1,
#     "gpt4all-13b-snoozy": 0.1,
#     "koala-13b": 0.1,
#     "stablelm-tuned-alpha-7b": 0.1,
#     "dolly-v2-12b": 0.1,
#     "llama-13b": 0.1,
#     "chatglm-6b": 0.5,
#     "deluxe-chat-v1": 4,
#     "palm-2": 1.5,
# }

# # target model sampling weights will be boosted.
# BATTLE_TARGETS = {
#     "gpt-4": {"gpt-4-0314", "claude-2.1", "gpt-4-turbo"},
#     "gpt-4-0613": {"gpt-4-0314", "claude-2.1", "gpt-4-turbo"},
#     "gpt-4-0314": {"gpt-4-turbo", "gpt-4-0613", "claude-2.1", "gpt-3.5-turbo-0613"},
#     "gpt-4-turbo": {
#         "gpt-4-0613",
#         "gpt-3.5-turbo-0613",
#         "gpt-3.5-turbo-1106",
#         "claude-2.1",
#     },
#     "gpt-3.5-turbo-0613": {"claude-instant-1", "gpt-4-0613", "claude-2.1"},
#     "gpt-3.5-turbo-1106": {"gpt-4-0613", "claude-instant-1", "gpt-3.5-turbo-0613"},
#     "solar-10.7b-instruct-v1.0": {
#         "mixtral-8x7b-instruct-v0.1",
#         "gpt-3.5-turbo-0613",
#         "llama-2-70b-chat",
#     },
#     "mixtral-8x7b-instruct-v0.1": {
#         "gpt-3.5-turbo-1106",
#         "gpt-3.5-turbo-0613",
#         "gpt-4-turbo",
#         "llama-2-70b-chat",
#     },
#     "claude-2.1": {"gpt-4-turbo", "gpt-4-0613", "claude-1"},
#     "claude-2.0": {"gpt-4-turbo", "gpt-4-0613", "claude-1"},
#     "claude-1": {"claude-2.1", "gpt-4-0613", "gpt-3.5-turbo-0613"},
#     "claude-instant-1": {"gpt-3.5-turbo-1106", "claude-2.1"},
#     "gemini-pro": {"gpt-4-turbo", "gpt-4-0613", "gpt-3.5-turbo-0613"},
#     "deluxe-chat-v1.1": {"gpt-4-0613", "gpt-4-turbo"},
#     "deluxe-chat-v1.2": {"gpt-4-0613", "gpt-4-turbo"},
#     "pplx-7b-online": {"gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "llama-2-70b-chat"},
#     "pplx-70b-online": {"gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "llama-2-70b-chat"},
#     "openhermes-2.5-mistral-7b": {
#         "gpt-3.5-turbo-0613",
#         "openchat-3.5",
#         "zephyr-7b-beta",
#     },
#     "dolphin-2.2.1-mistral-7b": {
#         "gpt-3.5-turbo-0613",
#         "vicuna-33b",
#         "starling-lm-7b-alpha",
#         "openhermes-2.5-mistral-7b",
#     },
#     "starling-lm-7b-alpha": {"gpt-3.5-turbo-0613", "openchat-3.5", "zephyr-7b-beta"},
#     "tulu-2-dpo-70b": {"gpt-3.5-turbo-0613", "vicuna-33b", "claude-instant-1"},
#     "yi-34b-chat": {"gpt-3.5-turbo-0613", "vicuna-33b", "claude-instant-1"},
#     "openchat-3.5": {"gpt-3.5-turbo-0613", "llama-2-70b-chat", "zephyr-7b-beta"},
#     "chatglm3-6b": {"yi-34b-chat", "qwen-14b-chat"},
#     "qwen-14b-chat": {"vicuna-13b", "llama-2-13b-chat", "llama-2-70b-chat"},
#     "zephyr-7b-alpha": {"mistral-7b-instruct", "llama-2-13b-chat"},
#     "zephyr-7b-beta": {
#         "mistral-7b-instruct",
#         "llama-2-13b-chat",
#         "llama-2-7b-chat",
#         "wizardlm-13b",
#     },
#     "llama-2-70b-chat": {"gpt-3.5-turbo-0613", "vicuna-33b", "claude-instant-1"},
#     "llama-2-13b-chat": {"mistral-7b-instruct", "vicuna-13b", "llama-2-70b-chat"},
#     "llama-2-7b-chat": {"mistral-7b-instruct", "vicuna-7b", "llama-2-13b-chat"},
#     "mistral-7b-instruct": {
#         "llama-2-7b-chat",
#         "llama-2-13b-chat",
#         "llama-2-70b-chat",
#     },
#     "vicuna-33b": {"llama-2-70b-chat", "gpt-3.5-turbo-0613", "claude-instant-1"},
#     "vicuna-13b": {"llama-2-13b-chat", "llama-2-70b-chat"},
#     "vicuna-7b": {"llama-2-7b-chat", "mistral-7b-instruct", "llama-2-13b-chat"},
#     "wizardlm-70b": {"gpt-3.5-turbo-0613", "vicuna-33b", "claude-instant-1"},
# }

# SAMPLING_BOOST_MODELS = [
#     # "tulu-2-dpo-70b",
#     # "yi-34b-chat",
#     "claude-2.1",
#     "claude-1",
#     "gpt-4-0613",
#     # "gpt-3.5-turbo-1106",
#     # "gpt-4-0314",
#     "gpt-4-turbo",
#     # "dolphin-2.2.1-mistral-7b",
#     "mixtral-8x7b-instruct-v0.1",
#     "gemini-pro",
#     "solar-10.7b-instruct-v1.0",
# ]

# DOWNLOAD_DATASET = os.getenv("DOWNLOAD_DATASET", "VISITBENCH,TOUCHSTONE")
DOWNLOAD_DATASET = os.getenv("DOWNLOAD_DATASET", "NA")
from datasets import load_dataset, concatenate_datasets
# subset_category = random.choice(['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'])
# mmmu_datasets = load_dataset("MMMU/MMMU", subset_category, split="validation")
# sample_examle = get_random_examples_mmmu(mmmu_datasets, num_examples=1)
VISITBENCH_DATASETS = load_dataset("mlfoundations/VisIT-Bench", split="test") if "VISITBENCH" in DOWNLOAD_DATASET else None
# WILDVISIONBENCH_DATASETS = load_dataset("WildVision/wildvision-bench", split="test")
TOUCHSTONE_DATASETS = load_dataset("WildVision/PublicBenchHub", "touchstone", split="test") if "TOUCHSTONE" in DOWNLOAD_DATASET else None
# FIXME: merge logic, to align the column names
# SAMPLE_DATASETS = concatenate_datasets([VISITBENCH_DATASETS, TOUCHSTONE_DATASET])

CONTROLLER_HEART_BEAT_EXPIRATION = int(
    os.getenv("FASTCHAT_CONTROLLER_HEART_BEAT_EXPIRATION", 90)
)
WORKER_HEART_BEAT_INTERVAL = int(os.getenv("FASTCHAT_WORKER_HEART_BEAT_INTERVAL", 45))
WORKER_API_TIMEOUT = int(os.getenv("FASTCHAT_WORKER_API_TIMEOUT", 100))
WORKER_API_EMBEDDING_BATCH_SIZE = int(
    os.getenv("FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE", 4)
)


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006
