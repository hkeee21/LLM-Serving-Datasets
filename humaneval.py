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
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:


    from human_eval.data import write_jsonl, read_problems
    import random
    problems = read_problems()
    samples = [
        problems[task_id]["prompt"]
        for task_id in problems
        # for i in range(20)
    ]
    import random
    random.seed(0)

    input_len_list = []
    output_len_list = []
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt_data in samples:
        pred = tokenizer.encode(prompt_data)
        output_len = random.randint(128, 256)
        # if len(pred) + output_len > 4096:
        #     continue
        input_len_list.append(len(pred))
        output_len_list.append(output_len)
    # import random
    # random.seed(777)
    # some of these will be filtered out, so sample more than we need
    # sampled_indices = random.sample(range(len(dataset)),
    #                                 int(num_requests * 1.2))
    # dataset = [dataset[i] for i in sampled_indices]
    # dataset = random.sample(dataset, num_requests * 3)

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
    input_requests = sample_requests(tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset Distribution.")
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
