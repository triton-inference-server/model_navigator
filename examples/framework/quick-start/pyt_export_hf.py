# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from pathlib import Path

import pandas as pd
from tabulate import tabulate

import model_navigator as nav

LOGGER = logging.getLogger()


def get_args():
    parser = argparse.ArgumentParser(description="Export huggingface models.")
    parser.add_argument("--workspace", type=str, default="./", help="Workspace path")
    parser.add_argument("--report-file", default="report.csv", type=str, help="Report file name (csv)")
    parser.add_argument("--max-sequence-len", default=None, type=int)
    parser.add_argument("--max-batch-size", default=1, type=int)
    return parser.parse_args()


def get_model_names():
    return [
        "distilbert-base-uncased",
        "gpt2",
        "bert-base-uncased",
        "distilgpt2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "bert-base-chinese",
        # "cardiffnlp/twitter-roberta-base-sentiment",
        # "deepset/roberta-base-squad2",
        # "roberta-base",
    ]


if __name__ == "__main__":
    args = get_args()
    model_names = get_model_names()

    results = []
    for model_name in model_names:
        LOGGER.info(f"Exporting {model_name}...")
        nav_workdir = Path(args.workspace) / "navigator_workdir"
        artifacts_library = nav.contrib.huggingface.torch.export(
            model_name=model_name,
            dataset_name="imdb",
            padding="max_length",
            max_sequence_len=args.max_sequence_len,
            max_bs=args.max_batch_size,
            sample_count=10,
            override_workdir=True,
            workdir=nav_workdir,
        )

        data = artifacts_library.get_formats_status()
        perf_status = artifacts_library.get_formats_performance()
        for format, perf in perf_status.items():
            if perf is not None:
                data[format] = perf[-1].latency
        results.append({"model_name": model_name, "max_sequence_len": args.max_sequence_len, **data})

    df = pd.DataFrame(results)
    print(tabulate(df, headers="keys", tablefmt="psql"))
    df.to_csv(args.report_file, index=False)
