"""
A model worker that executes the model.
"""
import argparse
import base64
import gc
import json
import os
from typing import List, Optional
import uuid

import torch
import torch.nn.functional as F
from transformers import set_seed
import uvicorn

from .constants import ErrorCode, SERVER_ERROR_MSG
from .model.model_adapter import (
    load_model,
    load_llava_pretrained_model,
    load_qwen_pretrained_model,
    load_blip_pretrained_model,
    load_uform_pretrained_model,
    load_idefics_pretrained_model,
    load_tinyllava_pretrained_model,
    load_deepseekvl_pretrained_model,
    load_bunny_pretrained_model,
    load_yivl_pretrained_model,
    add_model_args,
    get_generate_stream_function,
)

from .base_model_worker import BaseModelWorker, app
from .utils import (
    build_logger,
    get_context_length,
    str_to_torch_dtype,
)

worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")


class ModelWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        revision: str = None,
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        stream_interval: int = 2,
        conv_template: Optional[str] = None,
        embed_in_truncate: bool = False,
        seed: Optional[int] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.is_llavav15_stream = "llava-v1.5" in model_path.lower()
        self.is_llava_stream = "llava-v1.6" in model_path.lower()
        self.is_qwenvl_stream = "qwen-vl-chat" in model_path.lower()
        self.is_blip_stream = "blip" in model_path.lower()
        self.is_uform_stream = "uform" in model_path.lower()
        self.is_tinyllava_stream = "tiny-llava" in model_path.lower()
        self.is_deepseekvl_stream = "deepseek-vl" in model_path.lower()
        self.is_bunny_stream = "bunny" in model_path.lower()
        self.is_yivl_stream = "yi-vl" in model_path.lower() and not "plus" in model_path.lower()
        self.is_idefics_stream = "idefics2-local" in model_path.lower()
        self.is_videollava_stream = "video-llava" in model_path.lower()
        self.is_llavanext_stream = "llava-next" in model_path.lower()
        self.is_videollama2_stream = "videollama2" in model_path.lower()

        if self.is_llava_stream or self.is_llavav15_stream:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_llava_pretrained_model(
                model_path,
                device=device,
                load_8bit=load_8bit,          
            )
        elif self.is_qwenvl_stream:
            self.tokenizer, self.model, self.dtype = load_qwen_pretrained_model(
                model_path,
                device=device,
                dtype=dtype,       
            )
        elif self.is_blip_stream:
            self.model, self.image_processor = load_blip_pretrained_model(
                model_path,
                device=device,    
            )
        elif self.is_uform_stream:
            self.model, self.image_processor = load_uform_pretrained_model(
                model_path,
                device=device,          
            )
        elif self.is_idefics_stream:
            self.model, self.image_processor = load_idefics_pretrained_model(
                model_path,
                device=device,            
            )
        elif self.is_tinyllava_stream:
            self.model = load_tinyllava_pretrained_model(
                model_path,
                device=device,              
            )
        elif self.is_deepseekvl_stream:
            self.model, self.tokenizer, self.image_processor = load_deepseekvl_pretrained_model(
                model_path,           
            )
        elif self.is_bunny_stream:
            self.model, self.tokenizer = load_bunny_pretrained_model(
                model_path,           
            )
        elif self.is_yivl_stream:
            self.model, self.tokenizer, self.image_processor = load_yivl_pretrained_model(
                model_path,               
            )
        elif self.is_videollava_stream:
            # model_path = 'LanguageBind/Video-LLaVA-7B'
            cache_dir = 'cache_dir'
            device = 'cuda'
            load_4bit, load_8bit = True, False
            from .model.vlm_utils.videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
            model_name = get_model_name_from_path(model_path)
            from .model.vlm_utils.videollava.model.builder import load_pretrained_model
            self.tokenizer, self.model, self.processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
        elif self.is_llavanext_stream:
            cache_dir = 'cache_dir'
            device = 'cuda'
            load_4bit, load_8bit = True, False
            from .model.vlm_utils.llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
            model_name = get_model_name_from_path(model_path)
            from .model.vlm_utils.llavavid.model.builder import load_pretrained_model
            self.tokenizer, self.model, self.processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit)
        elif self.is_videollama2_stream:
            cache_dir = 'cache_dir'
            device = 'cuda'
            load_4bit, load_8bit = True, False
            from .model.vlm_utils.videollama2.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
            model_name = get_model_name_from_path(model_path)
            from .model.vlm_utils.videollama2.model.builder import load_pretrained_model
            self.tokenizer, self.model, self.processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit)
        else:
            self.model, self.tokenizer = load_model(
                model_path,
                revision=revision,
                device=device,
                num_gpus=num_gpus,
                max_gpu_memory=max_gpu_memory,
                dtype=dtype,
                load_8bit=load_8bit,
                cpu_offloading=cpu_offloading,
                debug=debug,
            )
        

        self.device = device
        if self.tokenizer is not None and self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.context_len = get_context_length(self.model.config)
        except:
            self.context_len = 1024
        # self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        self.generate_stream_func = get_generate_stream_function(model_path)
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            self.init_heart_beat()

    def generate_stream_gate(self, params):
        if self.device == "npu":
            import torch_npu

            torch_npu.npu.set_device("npu:0")
        self.call_ct += 1

        try:
            if self.seed is not None:
                set_seed(self.seed)
            if self.is_llava_stream or self.is_llavav15_stream or self.is_deepseekvl_stream or self.is_yivl_stream:
                for output in self.generate_stream_func(
                    self.model,
                    self.tokenizer,
                    self.image_processor,
                    params,
                    self.device,
                    self.context_len,
                    self.stream_interval,
                ):
                    # from icecream import ic
                    # ic(output, type(output))
                    ret = {
                        "text": output["text"],
                        "error_code": 0,
                    }
                    if "usage" in output:
                        ret["usage"] = output["usage"]
                    if "finish_reason" in output:
                        ret["finish_reason"] = output["finish_reason"]
                    if "logprobs" in output:
                        ret["logprobs"] = output["logprobs"]
                    yield json.dumps(ret).encode() + b"\0"      
            elif self.is_qwenvl_stream or self.is_bunny_stream:
                for output in self.generate_stream_func(
                    self.model,
                    self.tokenizer,
                    None,
                    params,
                    self.device,
                    -1,
                    self.stream_interval,
                ):
                    ret = {
                        "text": output["text"],
                        "error_code": 0,
                    }
                    if "usage" in output:
                        ret["usage"] = output["usage"]
                    if "finish_reason" in output:
                        ret["finish_reason"] = output["finish_reason"]
                    if "logprobs" in output:
                        ret["logprobs"] = output["logprobs"]
                    yield json.dumps(ret).encode() + b"\0"  
            elif self.is_blip_stream or self.is_uform_stream or self.is_idefics_stream:
                for output in self.generate_stream_func(
                    self.model,
                    None,
                    self.image_processor,
                    params,
                    self.device,
                    -1,
                    self.stream_interval,
                ):
                    ret = {
                        "text": output["text"],
                        "error_code": 0,
                    }
                    if "usage" in output:
                        ret["usage"] = output["usage"]
                    if "finish_reason" in output:
                        ret["finish_reason"] = output["finish_reason"]
                    if "logprobs" in output:
                        ret["logprobs"] = output["logprobs"]
                    yield json.dumps(ret).encode() + b"\0"   
            elif self.is_tinyllava_stream:
                for output in self.generate_stream_func(
                    self.model,
                    None,
                    None,
                    params,
                    self.device,
                    -1,
                    self.stream_interval,
                ):
                    ret = {
                        "text": output["text"],
                        "error_code": 0,
                    }
                    if "usage" in output:
                        ret["usage"] = output["usage"]
                    if "finish_reason" in output:
                        ret["finish_reason"] = output["finish_reason"]
                    if "logprobs" in output:
                        ret["logprobs"] = output["logprobs"]
                    yield json.dumps(ret).encode() + b"\0"
            elif self.is_videollava_stream:
                for output in self.generate_stream_func(
                    self.model,
                    self.tokenizer,
                    self.processor,
                    params,
                    self.device,
                    self.context_len,
                    self.stream_interval,
                ):
                    # from icecream import ic
                    # ic(output, type(output))
                    ret = {
                        "text": output["text"],
                        "error_code": 0,
                    }
                    if "usage" in output:
                        ret["usage"] = output["usage"]
                    if "finish_reason" in output:
                        ret["finish_reason"] = output["finish_reason"]
                    if "logprobs" in output:
                        ret["logprobs"] = output["logprobs"]
                    yield json.dumps(ret).encode() + b"\0"  
            elif self.is_llavanext_stream:
                for output in self.generate_stream_func(
                    self.model,
                    self.tokenizer,
                    self.processor,
                    params,
                    self.device,
                    self.context_len,
                    self.stream_interval,
                ):
                    # from icecream import ic
                    # ic(output, type(output))
                    ret = {
                        "text": output["text"],
                        "error_code": 0,
                    }
                    if "usage" in output:
                        ret["usage"] = output["usage"]
                    if "finish_reason" in output:
                        ret["finish_reason"] = output["finish_reason"]
                    if "logprobs" in output:
                        ret["logprobs"] = output["logprobs"]
                    yield json.dumps(ret).encode() + b"\0" 
            elif self.is_videollama2_stream:
                for output in self.generate_stream_func(
                    self.model,
                    self.tokenizer,
                    self.processor,
                    params,
                    self.device,
                    self.context_len,
                    self.stream_interval,
                ):
                    # from icecream import ic
                    # ic(output, type(output))
                    ret = {
                        "text": output["text"],
                        "error_code": 0,
                    }
                    if "usage" in output:
                        ret["usage"] = output["usage"]
                    if "finish_reason" in output:
                        ret["finish_reason"] = output["finish_reason"]
                    if "logprobs" in output:
                        ret["logprobs"] = output["logprobs"]
                    yield json.dumps(ret).encode() + b"\0" 

            else:
                for output in self.generate_stream_func(
                    self.model,
                    self.tokenizer,
                    params,
                    self.device,
                    self.context_len,
                    self.stream_interval,
                ):
                    # from icecream import ic
                    # ic(output, type(output))
                    ret = {
                        "text": output["text"],
                        "error_code": 0,
                    }
                    if "usage" in output:
                        ret["usage"] = output["usage"]
                    if "finish_reason" in output:
                        ret["finish_reason"] = output["finish_reason"]
                    if "logprobs" in output:
                        ret["logprobs"] = output["logprobs"]
                    yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                # "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                # "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())

    def __process_embed_chunk(self, input_ids, attention_mask, **model_type_dict):
        if model_type_dict.get("is_bert"):
            model_output = self.model(input_ids)
            if model_type_dict.get("is_robert"):
                data = model_output.last_hidden_state
            else:
                data = model_output[0]
        elif model_type_dict.get("is_t5"):
            model_output = self.model(input_ids, decoder_input_ids=input_ids)
            data = model_output.encoder_last_hidden_state
        else:
            model_output = self.model(input_ids, output_hidden_states=True)
            if model_type_dict.get("is_chatglm"):
                data = model_output.hidden_states[-1].transpose(0, 1)
            else:
                data = model_output.hidden_states[-1]

        if hasattr(self.model, "use_cls_pooling") and self.model.use_cls_pooling:
            sum_embeddings = data[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
            masked_embeddings = data * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
        token_num = torch.sum(attention_mask).item()

        return sum_embeddings, token_num

    def __encode_base64(self, embeddings: torch.Tensor) -> List[str]:
        embeddings = embeddings.cpu()
        return [
            base64.b64encode(e.numpy().tobytes()).decode("utf-8") for e in embeddings
        ]

    @torch.inference_mode()
    def get_embeddings(self, params):
        self.call_ct += 1

        try:
            tokenizer = self.tokenizer
            ret = {"embedding": [], "token_num": 0}

            model_type_dict = {
                "is_llama": "llama" in str(type(self.model)),
                "is_t5": "t5" in str(type(self.model)),
                "is_chatglm": "chatglm" in str(type(self.model)),
                "is_bert": "bert" in str(type(self.model)),
                "is_robert": "robert" in str(type(self.model)),
            }

            if self.embed_in_truncate:
                encoding = tokenizer.batch_encode_plus(
                    params["input"],
                    padding=True,
                    truncation="longest_first",
                    return_tensors="pt",
                    max_length=self.context_len,
                )
            else:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = input_ids != tokenizer.pad_token_id

            base64_encode = params.get("encoding_format", None)

            if self.embed_in_truncate:
                embedding, token_num = self.__process_embed_chunk(
                    input_ids, attention_mask, **model_type_dict
                )
                if (
                    not hasattr(self.model, "use_cls_pooling")
                    or not self.model.use_cls_pooling
                ):
                    embedding = embedding / token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret["token_num"] = token_num
            else:
                all_embeddings = []
                all_token_num = 0
                for i in range(0, input_ids.size(1), self.context_len):
                    chunk_input_ids = input_ids[:, i : i + self.context_len]
                    chunk_attention_mask = attention_mask[:, i : i + self.context_len]

                    # add cls token and mask to get cls embedding
                    if (
                        hasattr(self.model, "use_cls_pooling")
                        and self.model.use_cls_pooling
                    ):
                        cls_tokens = (
                            torch.zeros(
                                (chunk_input_ids.size(0), 1),
                                dtype=chunk_input_ids.dtype,
                                device=chunk_input_ids.device,
                            )
                            + tokenizer.cls_token_id
                        )
                        chunk_input_ids = torch.cat(
                            [cls_tokens, chunk_input_ids], dim=-1
                        )
                        mask = torch.ones(
                            (chunk_attention_mask.size(0), 1),
                            dtype=chunk_attention_mask.dtype,
                            device=chunk_attention_mask.device,
                        )
                        chunk_attention_mask = torch.cat(
                            [mask, chunk_attention_mask], dim=-1
                        )

                    chunk_embeddings, token_num = self.__process_embed_chunk(
                        chunk_input_ids, chunk_attention_mask, **model_type_dict
                    )
                    if (
                        hasattr(self.model, "use_cls_pooling")
                        and self.model.use_cls_pooling
                    ):
                        all_embeddings.append(chunk_embeddings * token_num)
                    else:
                        all_embeddings.append(chunk_embeddings)
                    all_token_num += token_num

                all_embeddings_tensor = torch.stack(all_embeddings)
                embedding = torch.sum(all_embeddings_tensor, dim=0) / all_token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)

                ret["token_num"] = all_token_num

            if base64_encode == "base64":
                out_embeddings = self.__encode_base64(normalized_embeddings)
            else:
                out_embeddings = normalized_embeddings.tolist()
            ret["embedding"] = out_embeddings

            gc.collect()
            torch.cuda.empty_cache()
            if self.device == "xpu":
                torch.xpu.empty_cache()
            if self.device == "npu":
                torch.npu.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                # "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                # "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret


def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--embed-in-truncate", action="store_true")
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Print debugging messages"
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        revision=args.revision,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        stream_interval=args.stream_interval,
        conv_template=args.conv_template,
        embed_in_truncate=args.embed_in_truncate,
        seed=args.seed,
        debug=args.debug,
    )
    return args, worker


if __name__ == "__main__":
    args, worker = create_model_worker()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
