import argparse
import asyncio
import json
import random
import time
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Tuple

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

def sample_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # import random
    # random.seed(777)
    # some of these will be filtered out, so sample more than we need
    # sampled_indices = random.sample(range(len(dataset)),
    #                                 int(num_requests * 1.2))
    # dataset = [dataset[i] for i in sampled_indices]
    # dataset = random.sample(dataset, num_requests * 3)

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # outlier_count = 0
    
    # Filter out too long sequences.
    # filtered_dataset: List[Tuple[str, int, int]] = []
    input_len_list = []
    output_len_list = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        # filtered_dataset.append((prompt, prompt_len, output_len))
        input_len_list.append(prompt_len)
        output_len_list.append(output_len)

    # Sample the requests.
    # sampled_requests = random.sample(filtered_dataset, num_requests)

    # sum_len = 0
    # for e in sampled_requests:
    #     sum_len += e[1] + e[2]
    # print("total tokens:", sum_len)
    # print("outliers:", outlier_count)
    # return sampled_requests

    # input mean, std, max; output mean, std, max
    print(f"{np.mean(input_len_list):.0f} {np.std(input_len_list):.0f} {np.max(input_len_list):.0f}, \
        {np.mean(output_len_list):.0f} {np.std(output_len_list):.0f} {np.max(output_len_list):.0f}")


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset Distribution.")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )

    args = parser.parse_args()
    main(args)
