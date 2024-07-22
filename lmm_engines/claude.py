import anthropic
import os
from typing import List
from anthropic import NOT_GIVEN

# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_claude(messages:List[dict], model_name, **generate_kwargs) -> str:
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
        generate_kwargs: additional keyword arguments for the generation
    """
    raise NotImplementedError("This function is not implemented yet")
    
if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_claude(["Hello", "Hi, I am claude", "What did I ask in the last response?"], "claude-3-opus-20240229"))