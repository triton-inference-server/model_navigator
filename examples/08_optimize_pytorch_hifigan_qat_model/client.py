#!/usr/bin/env python3
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
from pytriton.client import ModelClient  # pytype: disable=import-error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="tts_hifigan")
    parser.add_argument(
        "--server",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="audio",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with ModelClient(args.server, args.model_name) as client:
        input_dict = {"spec": np.random.randn(1, 80, 256).astype(np.float32)}

        results = client.infer_batch(**input_dict)

        np.save(args.output_path, results["output__0"])


if __name__ == "__main__":
    main()
