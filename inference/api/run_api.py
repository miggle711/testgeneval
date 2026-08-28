# Copyright (c) Meta Platforms, Inc. and affiliates.

"""This python script is designed to run inference on a dataset using the OpenAI API (including OpenAI-compatible local/hosted endpoints via LOCAL_MODEL_BASE_URL, e.g. local MLX, Groq, M3/vLLM).
It sorts instances by length and continually writes the outputs to a specified file, so that the script can be stopped and restarted without losing progress.
"""

import json
import logging
import os
import threading
import time
import traceback
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
import numpy as np
import openai
import tiktoken
from datasets import DatasetDict, load_dataset, load_from_disk
from frozendict import frozendict
from inference.configs.instruct_prompt import InstructPrompt
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
dotenv.load_dotenv()


class _KeyRotator:
    """Round-robins across multiple API keys for a custom OpenAI-compatible
    endpoint (LOCAL_MODEL_BASE_URL) -- e.g. several separate Groq accounts,
    to multiply effective per-minute throughput when one key's rate limit
    is too small for the request volume. Only active when
    LOCAL_MODEL_KEY_PREFIX is set; a single-key setup (LOCAL_MODEL_API_KEY)
    is unaffected. Does not help an individual request that's larger than
    any one key's per-request limit -- only spreads separate requests
    across more quota.
    """

    def __init__(self, prefix: str):
        self.keys = [
            v for k, v in sorted(os.environ.items())
            if k.endswith(prefix) and v
        ]
        self._i = 0

    def next(self):
        if not self.keys:
            return None
        key = self.keys[self._i % len(self.keys)]
        self._i += 1
        return key


_thread_local = threading.local()

_key_rotator = None
if os.environ.get("LOCAL_MODEL_KEY_PREFIX"):
    _key_rotator = _KeyRotator(os.environ["LOCAL_MODEL_KEY_PREFIX"])
    logger.info(f"Key rotation enabled: {len(_key_rotator.keys)} keys found "
                f"matching suffix '{os.environ['LOCAL_MODEL_KEY_PREFIX']}'")

# mlx-community/Meta-Llama-3.1-8B-Instruct-4bit: served locally (e.g. via an
# OpenAI-compatible local server), not a real OpenAI model -- 0 cost, and its
# context window/output limit follow the underlying Llama-3.1-8B model.
#
# llama-3.1-8b-instant: Groq-hosted (via LOCAL_MODEL_BASE_URL=
# https://api.groq.com/openai/v1/, LOCAL_MODEL_API_KEY=<real groq key>).
# Cost left at 0 -- Groq's real per-token pricing wasn't verified against a
# live source when this was added; update MODEL_COST_PER_INPUT/OUTPUT with
# real numbers from https://groq.com/pricing if accurate cost tracking
# matters. Context window/output limit are Groq's documented values for
# this model as of when this was added -- worth double-checking against
# Groq's docs if this starts erroring on token limits.
MODEL_LIMITS = {
    "gpt-3.5-turbo-0125": 16_385,
    "gpt-4-turbo-2024-04-09": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "gpt-4-0613": 8_192,
    "Meta-Llama-3.1-405B-Instruct": 128_000,
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": 128_000,
    "llama-3.1-8b-instant": 128_000,
}

# The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    "gpt-3.5-turbo-0125": 0.0000005,
    "gpt-4-turbo-2024-04-09": 0.00001,
    "gpt-4o-2024-05-13": 0.000005,
    "gpt-4-0613": 0.00001,
    "Meta-Llama-3.1-405B-Instruct": 0,
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": 0,
    "llama-3.1-8b-instant": 0,
}

# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    "gpt-3.5-turbo-0125": 0.0000015,
    "gpt-4-turbo-2024-04-09": 0.00003,
    "gpt-4o-2024-05-13": 0.000015,
    "gpt-4-0613": 0.00003,
    "Meta-Llama-3.1-405B-Instruct": 0,
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": 0,
    "llama-3.1-8b-instant": 0,
}

OUTPUT_LIMITS = {
    "gpt-3.5-turbo-0125": 4_096,
    "gpt-4-turbo-2024-04-09": 8_192,
    "gpt-4o-2024-05-13": 4_096,
    "gpt-4-0613": 8_192,
    "Meta-Llama-3.1-405B-Instruct": 4_096,
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": 4_096,
    "llama-3.1-8b-instant": 8_192,
}

EPSILON = 1000


def calc_cost(model_name, input_tokens, output_tokens):
    """
    Calculates the cost of a response from the openai API.

    Args:
    response (openai.ChatCompletion): The response from the API.

    Returns:
    float: The cost of the response.
    """
    # .get(..., 0): an unregistered model (any model served via
    # LOCAL_MODEL_BASE_URL that hasn't been added to these dicts -- e.g.
    # each of the ~8 planned M3 models) has no real per-token pricing here,
    # so cost is 0 rather than a KeyError. Add a real entry if accurate
    # cost tracking matters for that model.
    cost = (
        MODEL_COST_PER_INPUT.get(model_name, 0) * input_tokens
        + MODEL_COST_PER_OUTPUT.get(model_name, 0) * output_tokens
    )
    logger.info(
        f"input_tokens={input_tokens}, output_tokens={output_tokens}, cost={cost:.2f}"
    )
    return cost


@retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
def call_chat(
    model_name_or_path,
    inputs,
    temperature,
    top_p,
    max_tokens,
    system_message,
    no_system_message=False,
    n=1,
    **model_args,
):
    """
    Calls the OpenAI API to generate completions for the given inputs using the new API interface.

    Args:
        model_name_or_path (str): The name or path of the model to use.
        inputs (str): The inputs to generate completions for.
        temperature (float): The temperature to use.
        top_p (float): The top_p to use.
        no_system_message (bool): Fold system_message into the user turn
            instead of sending a separate system role. Some models' chat
            templates reject a system role entirely (confirmed real:
            bigcode/starcoder2-15b-instruct-v0.1 400s on every request
            with "System messages are not allowed in this template").
        n (int): Number of independent completions to request in this one
            call, via the OpenAI API's own n parameter (supported by
            vLLM's server too). One prompt prefill, n samples, instead of
            n separate calls each repaying prefill cost. Only meaningful
            when temperature > 0; requesting n > 1 at temperature 0 would
            just return n identical completions.
        **model_args (dict): Additional model arguments.

    Returns:
        tuple: A tuple containing the response and the cost of the completion.
    """
    user_message = inputs
    if no_system_message:
        messages = [
            {"role": "user", "content": f"{system_message}\n\n{user_message}"},
        ]
    else:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    # A per-thread client (rather than mutating the shared openai.api_key/
    # base_url module globals) so concurrent requests under a
    # ThreadPoolExecutor can't race and send one thread's rotated key on
    # another thread's request. Cached on _thread_local so each worker
    # thread builds its client once, not per-request -- the key-rotator
    # path still needs a fresh client per call since the key can change
    # call to call.
    if _key_rotator is not None:
        next_key = _key_rotator.next()
        client = openai.OpenAI(
            api_key=next_key or openai.api_key, base_url=openai.base_url
        )
    else:
        if not hasattr(_thread_local, "client"):
            _thread_local.client = openai.OpenAI(
                api_key=openai.api_key, base_url=openai.base_url
            )
        client = _thread_local.client

    try:
        response = client.chat.completions.create(
            model=model_name_or_path,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,  # Adjust max_tokens as needed
            top_p=top_p,
            n=n,
            **model_args,
        )

        input_tokens = response.usage.prompt_tokens
        # completion_tokens is the sum across all n completions, not
        # per-completion -- that's the real cost incurred, so calc_cost
        # doesn't need adjusting for n > 1.
        output_tokens = response.usage.completion_tokens
        cost = calc_cost(model_name_or_path, input_tokens, output_tokens)
        return response, cost

    except Exception as e:
        print(f"API Error: {e}")
        raise


def gpt_tokenize(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens


def openai_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    max_cost,
    num_samples,
    postprocess_fn,
    system_message,
    system_message_full,
    skip_full,
    skip_completion=False,
    model_nickname=None,
    no_system_message=False,
    max_concurrency=1,
):
    """
    Runs inference on a dataset using the openai API.

    Args:
    test_dataset (datasets.Dataset): The dataset to run inference on.
    model_name_or_path (str): The name or path of the model to use.
    output_file (str): The path to the output file.
    model_args (dict): A dictionary of model arguments.
    existing_ids (set): A set of ids that have already been processed.
    max_cost (float): The maximum cost to spend on inference.
    num_samples (int): The number of samples to generate for each prompt.
    model_nickname (str): Slash-free name recorded as the eval-time model
        id (basic_args["model_name_or_path"]) -- the container-side eval
        code embeds this raw into a log filename
        (f"{id}.{model}.{setting}.eval.log"), so a model_name_or_path with
        a "/" (e.g. "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit") breaks
        that path. The real model_name_or_path is still used for the
        actual API calls below. Defaults to model_name_or_path when there's
        no "/" to strip.
    """
    # tiktoken only knows real OpenAI model names -- a locally served model
    # (e.g. via LOCAL_MODEL_BASE_URL) falls back to gpt-4's encoding, which
    # is a reasonable approximation for token-count-based truncation, not an
    # exact match for the local model's own tokenizer.
    try:
        encoding = tiktoken.encoding_for_model(model_name_or_path)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4")
    # .get(..., default): an unregistered model served via
    # LOCAL_MODEL_BASE_URL (e.g. one of the ~8 planned M3 models before
    # it's added to MODEL_LIMITS/OUTPUT_LIMITS) falls back to a
    # conservative default rather than a KeyError. Add real entries once a
    # model's actual context window/output limit are known.
    default_output_limit = OUTPUT_LIMITS.get(model_name_or_path, 4_096)
    model_limit = (
        MODEL_LIMITS.get(model_name_or_path, 32_000) - default_output_limit - EPSILON
    )

    # Adjust dataset to truncate prompts to the last model_limit tokens
    def truncate_prompts(example):
        truncated = {}
        for key, prompt in example["preds_prompts"].items():
            tokenized = encoding.encode(prompt)
            if len(tokenized) > model_limit:
                # Truncate to the last model_limit tokens and decode back to text
                truncated[key] = encoding.decode(tokenized[-model_limit:])
            else:
                truncated[key] = prompt
        example["preds_prompts"] = truncated
        return example

    test_dataset = test_dataset.map(truncate_prompts, load_from_cache_file=False)

    # LOCAL_MODEL_BASE_URL (e.g. http://127.0.0.1:8000/v1/ for a local
    # server, or https://api.groq.com/openai/v1/ for Groq: trailing slash
    # required, or the openai client concatenates the path without a
    # separator and every request 404s) points the openai client at any
    # OpenAI-compatible endpoint instead of api.openai.com. Despite the
    # name, this also covers real hosted providers like Groq, not just a
    # local server -- LOCAL_MODEL_API_KEY supplies the real key for those;
    # a genuinely local, unauthenticated server can leave it unset.
    local_base_url = os.environ.get("LOCAL_MODEL_BASE_URL")
    if local_base_url:
        openai.base_url = local_base_url
        openai.api_key = os.environ.get("LOCAL_MODEL_API_KEY", "not-needed")
        print(f"Using custom OpenAI-compatible endpoint at {local_base_url}")
    else:
        openai_key = os.environ.get("OPENAI_API_KEY", None)
        if openai_key is None:
            raise ValueError(
                "Must provide an api key. Expected in OPENAI_API_KEY environment variable."
            )
        openai.api_key = openai_key
        print(f"Using OpenAI key {'*' * max(0, len(openai_key)-5) + openai_key[-5:]}")
    print(model_args)
    temperature = model_args.pop("temperature", 0.2)
    top_p = model_args.pop("top_p", 0.95 if temperature > 0 else 1)
    print(f"Using temperature={temperature}, top_p={top_p}")
    recorded_name = model_nickname if model_nickname else model_name_or_path
    basic_args = {
        "model_name_or_path": recorded_name + f"_t={temperature}",
    }
    print(f"Filtered to {len(test_dataset)} instances")

    # Shared across worker threads: a lock around both, since
    # ThreadPoolExecutor runs process_instance concurrently for
    # max_concurrency > 1. max_cost is a soft stop -- in-flight requests
    # aren't cancelled, new ones just stop being submitted once it's hit.
    cost_lock = threading.Lock()
    state = {"total_cost": 0, "cost_exceeded": False}
    write_lock = threading.Lock()

    def process_instance(datum):
        curr_id = datum["id"]
        if curr_id in existing_ids:
            return None
        with cost_lock:
            if state["cost_exceeded"]:
                return None
        output_dict = {"id": curr_id, "instance_id": datum["instance_id"]}
        output_dict.update(basic_args)
        output_dict["preds_prompts"] = datum["preds_prompts"]
        output_dict["preds"] = {}
        failed = False
        for prompt_name, prompt_text in datum["preds_prompts"].items():
            prompt_predictions = []
            if skip_full and prompt_name == "full":
                continue
            if skip_completion and prompt_name != "full":
                continue
            if prompt_name == "full":
                # One call requesting num_samples completions via the
                # API's own n parameter, instead of num_samples separate
                # calls -- one prompt prefill instead of num_samples of
                # them. Previously hardcoded to 1 sample regardless of
                # --num_samples for this setting; num_samples now applies
                # here too, needed for pass@k evaluation (k=num_samples).
                try:
                    response, cost = call_chat(
                        model_name_or_path,
                        prompt_text,
                        temperature,
                        top_p,
                        default_output_limit,
                        system_message_full,
                        no_system_message=no_system_message,
                        n=num_samples,
                    )
                    for choice in response.choices:
                        prompt_predictions.append(
                            postprocess_fn(choice.message.content, True)
                        )
                    with cost_lock:
                        state["total_cost"] += cost
                        if max_cost is not None and state["total_cost"] >= max_cost:
                            print(f"Reached max cost {max_cost}, exiting")
                            state["cost_exceeded"] = True
                except Exception as e:
                    print(f"Error: {e}")
                    failed = True
            else:
                for _ in range(num_samples):
                    try:
                        response, cost = call_chat(
                            model_name_or_path,
                            prompt_text,
                            temperature,
                            top_p,
                            512,
                            system_message,
                            no_system_message=no_system_message,
                        )
                        completion = response.choices[0].message.content
                        prompt_predictions.append(
                            postprocess_fn(completion, False)
                        )
                        with cost_lock:
                            state["total_cost"] += cost
                            if max_cost is not None and state["total_cost"] >= max_cost:
                                print(f"Reached max cost {max_cost}, exiting")
                                state["cost_exceeded"] = True
                    except Exception as e:
                        print(f"Error: {e}")
                        failed = True
            output_dict["preds"][prompt_name] = prompt_predictions
        if failed:
            print("Failed, skipping...")
            return None
        return output_dict

    with open(output_file, "a+") as f:
        if max_concurrency <= 1:
            for datum in tqdm(
                test_dataset, desc=f"Inference for {model_name_or_path}"
            ):
                result = process_instance(datum)
                if result is not None:
                    print(json.dumps(result), file=f, flush=True)
                if state["cost_exceeded"]:
                    return
        else:
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = {
                    executor.submit(process_instance, datum): datum
                    for datum in test_dataset
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Inference for {model_name_or_path}",
                ):
                    result = future.result()
                    if result is not None:
                        with write_lock:
                            print(json.dumps(result), file=f, flush=True)
        print(f"Total Cost: {state['total_cost']:.2f}")



def parse_model_args(model_args):
    """
    Parses a string of model arguments and returns a dictionary of keyword arguments.

    Args:
        model_args (str): A string of comma-separated key-value pairs representing model arguments.

    Returns:
        dict: A dictionary of keyword arguments parsed from the input string.
    """
    kwargs = dict()
    if model_args is not None:
        for arg in model_args.split(","):
            key, value = arg.split("=")
            # infer value type
            if value in {"True", "False"}:
                kwargs[key] = value == "True"
            elif value.isnumeric():
                kwargs[key] = int(value)
            elif value.replace(".", "", 1).isnumeric():
                kwargs[key] = float(value)
            elif value in {"None"}:
                kwargs[key] = None
            elif value in {"[]"}:
                kwargs[key] = []
            elif value in {"{}"}:
                kwargs[key] = {}
            elif value.startswith("'") and value.endswith("'"):
                kwargs[key] = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                kwargs[key] = value[1:-1]
            else:
                kwargs[key] = value
    return kwargs


def main(
    dataset_name_or_path,
    split,
    model_name_or_path,
    shard_id,
    num_shards,
    output_dir,
    no_imports,
    model_args,
    max_cost,
    num_samples,
    skip_full,
    skip_completion=False,
    prompt_config="instruct",
    kg_prompts_path="kg_prompts.json",
    no_system_message=False,
    max_concurrency=1,
):
    if shard_id is None and num_shards is not None:
        logger.warning(
            f"Received num_shards={num_shards} but shard_id is None, ignoring"
        )
    if shard_id is not None and num_shards is None:
        logger.warning(f"Received shard_id={shard_id} but num_shards is None, ignoring")
    model_args = parse_model_args(model_args)

    if prompt_config == "kg_only":
        from inference.configs.kg_only_prompt import KGOnlyPrompt
        prompt_info = KGOnlyPrompt(prompts_path=kg_prompts_path)
    else:
        # Also reads kg_prompts_path for target_functions/target_classes,
        # so instruct focuses on the same changed function(s) kg_only
        # does. Falls back to unfocused prompts if the file doesn't
        # exist -- logged loudly, since os.path.exists is resolved
        # against whatever CWD this process happens to run from (e.g.
        # via subprocess from run_pipeline.py or the M3 slurm script),
        # so a real file elsewhere can silently miss with no other signal.
        if os.path.exists(kg_prompts_path):
            resolved_kg_prompts_path = kg_prompts_path
        else:
            resolved_kg_prompts_path = None
            logger.warning(
                f"kg_prompts_path={kg_prompts_path!r} not found relative to "
                f"cwd={os.getcwd()!r} -- instruct will use unfocused prompts "
                f"(no target function/class wording). If this file exists "
                f"elsewhere, pass an absolute path."
            )
        prompt_info = InstructPrompt(kg_prompts_path=resolved_kg_prompts_path)

    model_nickname = model_name_or_path
    if "checkpoint" in Path(model_name_or_path).name:
        model_nickname = Path(model_name_or_path).parent.name
    else:
        model_nickname = Path(model_name_or_path).name

    temperature = model_args["temperature"] if "temperature" in model_args else 0.2
    output_file = f"{model_nickname}__{dataset_name_or_path.split('/')[-1]}__{temperature}__{split}"
    if shard_id is not None and num_shards is not None:
        output_file += f"__shard-{shard_id}__num_shards-{num_shards}"
    output_file = Path(output_dir, output_file + ".jsonl")
    logger.info(f"Will write to {output_file}")
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                curr_id = data["id"]
                existing_ids.add(curr_id)
    logger.info(f"Read {len(existing_ids)} already completed ids from {output_file}")
    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)

    dataset = prompt_info.add_prompts_to_dataset(dataset, no_import=no_imports)

    if not split in dataset:
        raise ValueError(f"Invalid split {split} for dataset {dataset_name_or_path}")
    dataset = dataset[split]

    print(dataset[0].keys())
    # Shard before filtering existing_ids, not after. Each shard's
    # output file can have a different number of already-completed
    # ids (resumed at a different point), so filtering first would
    # shrink the dataset by a different amount per shard, before
    # contiguous=True computes its slice, drifting shard boundaries
    # out of alignment and producing overlapping shards.
    if shard_id is not None and num_shards is not None:
        dataset = dataset.shard(num_shards, shard_id, contiguous=True)
    if len(existing_ids) > 0:
        dataset = dataset.filter(
            lambda x: x["id"] not in existing_ids,
            desc="Filtering out existing ids",
            load_from_cache_file=False,
        )
    inference_args = {
        "test_dataset": dataset,
        "model_name_or_path": model_name_or_path,
        "output_file": output_file,
        "model_args": model_args,
        "existing_ids": existing_ids,
        "max_cost": max_cost,
        "num_samples": num_samples,
        "postprocess_fn": prompt_info.postprocess_output,
        "system_message": prompt_info.system_message,
        "system_message_full": prompt_info.system_message_full,
        "skip_full": skip_full,
        "skip_completion": skip_completion,
    }
    if model_name_or_path.startswith("gpt") or os.environ.get("LOCAL_MODEL_BASE_URL"):
        # A model served locally via an OpenAI-compatible endpoint
        # (LOCAL_MODEL_BASE_URL set) uses the same call_chat/openai_inference
        # path real OpenAI models do -- it's the request shape that matters,
        # not the model name. model_nickname is the slash-free name (e.g.
        # a "mlx-community/..." model id) recorded as the eval-time model
        # id, since the container-side eval code embeds it raw into a log
        # filename.
        openai_inference(
            **inference_args,
            model_nickname=model_nickname,
            no_system_message=no_system_message,
            max_concurrency=max_concurrency,
        )
    else:
        raise ValueError(f"Invalid model name or path {model_name_or_path}")
    logger.info(f"Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name of API model, or a real HuggingFace model id for the "
             "local/M3 vLLM path. Unknown models fall back to a "
             "conservative default token limit (see MODEL_LIMITS.get).",
        default="gpt-3.5-turbo-1106",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard id to process. If None, process all shards.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards. If None, process all shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--no_imports",
        action="store_true",
        help="Use the no imports version of full.",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default=None,
        help="List of model arguments separated by commas. (e.g. 'top_p=0.95,temperature=0.70')",
    )
    parser.add_argument(
        "--max_cost",
        type=float,
        default=None,
        help="Maximum cost to spend on inference.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate for each prompt.",
    )
    parser.add_argument(
        "--skip_full",
        help="Whether to skip full setting.",
        action="store_true",
    )
    parser.add_argument(
        "--skip_completion",
        help="Whether to skip the completion settings (first/last/extra).",
        action="store_true",
    )
    parser.add_argument(
        "--prompt_config",
        type=str,
        choices=["instruct", "kg_only"],
        default="instruct",
        help="Which prompt strategy to use. 'instruct' (default) builds "
             "prompts from code_src/test_src ('full' setting only under "
             "the current test-generation scope). 'kg_only' reads "
             "pre-computed KG-derived prompts from --kg_prompts_path "
             "('full' setting only).",
    )
    parser.add_argument(
        "--kg_prompts_path",
        type=str,
        default="kg_prompts.json",
        help="Path to a JSON file of pre-computed KG prompts "
             "(scripts/build_kg_prompts.py in pycodekg). Required content "
             "for --prompt_config kg_only; also read by --prompt_config "
             "instruct for its target_function/target_class fields, so "
             "both arms are told to focus on the same changed function "
             "(miggle711/pycodekg#125, miggle711/testgeneval#6).",
    )
    parser.add_argument(
        "--no_system_message",
        action="store_true",
        help="Fold the system message into the user turn instead of "
             "sending a separate system role. Some models' chat templates "
             "reject a system role entirely (confirmed real: "
             "bigcode/starcoder2-15b-instruct-v0.1 400s on every request "
             "with 'System messages are not allowed in this template').",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=1,
        help="Number of instances to run inference on concurrently. "
             "Default 1 preserves the original one-request-at-a-time "
             "behavior. A local vLLM server has real headroom for more "
             "(continuous batching means multiple concurrent requests "
             "raise total throughput even though any one request's own "
             "latency is unchanged) -- 4-8 is a reasonable starting point "
             "on a single GPU node, watch vLLM's own GPU KV cache usage "
             "log line to see how much headroom is actually left.",
    )
    args = parser.parse_args()
    print(args.model_args)
    main(**vars(args))
