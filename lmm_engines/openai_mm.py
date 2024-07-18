from openai import OpenAI
from typing import List

# no image, multi-turn, do not use openai_generate, but can refer to it
def call_worker_openai(messages:List[dict], model_name, **generate_kwargs) -> str:
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
    # change messages to openai format
    new_messages = []
    if conv_system_msg:
        new_messages.append({"role": "system", "content": conv_system_msg})
    for i, message in enumerate(messages):
        new_messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": message})
    # initialize openai client
    client = OpenAI()
    # call openai
    response = client.chat.completions.create(
        model=model_name,
        messages=new_messages,
        **generate_kwargs,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    from icecream import ic
    ic(call_worker_openai(["Hello"], "gpt-3.5-turbo"))
    ic(call_worker_openai_completion("Hello", "gpt-3.5-turbo-instruct"))