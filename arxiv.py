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

    input_len_list = []
    output_len_list = []
    with open(dataset_path, encoding="utf-8") as f:
        for id_, row in enumerate(f):
            data = json.loads(row)
            # if id_ < 1:
            input_text = ""
            for c in data["article_text"]:
                input_text = input_text + c
            output_text = ""
            for c in data["abstract_text"]:
                output_text = output_text + c
            # input_text = tokenizer.encode([c for c in data["article_text"][0]])
            # output_text = tokenizer.encode(data["abstract_text"])
            input_len_list.append((len(input_text)))
            output_len_list.append(len(output_text))

    # import fastparquet as fp
    # parquet_file = fp.ParquetFile(dataset_path)
    # data = parquet_file.to_pandas()

    # # print(data)

    # ins_text = "Translate the following sentence to English."
    # pre_len = len(tokenizer.encode(ins_text))

    # input_len_list = []
    # output_len_list = []
    # for index, row in data.iterrows():
    #     # if index < 3:
    #     #     print(row["text"])
    #     input_text = tokenizer.encode(row["translation.cs"])
    #     output_text = tokenizer.encode(row["translation.en"])
    #     input_len_list.append((len(input_text) + pre_len))
    #     output_len_list.append(len(output_text))

    # input mean, std, max; output mean, std, max
    print(f"{np.mean(input_len_list):.0f} {np.std(input_len_list):.0f} {np.max(input_len_list):.0f} \
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
