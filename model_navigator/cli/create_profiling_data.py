#!/usr/bin/env python3

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from model_navigator.optimizer.config import parse_tensor_spec, parse_value_range
from model_navigator.perf_analyzer.profiling_data import create_profiling_data

LOGGER = logging.getLogger("profiling_data")


def main():
    parser = argparse.ArgumentParser(
        description="Create Triton model repository and model configuration", allow_abbrev=False
    )
    parser.add_argument("--shapes", nargs="+", required=True, help="List of input shapes.")
    parser.add_argument("--value-ranges", nargs="+", required=True, help="List of values ranges.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of dataloader iterations.")
    parser.add_argument("--output-path", type=str, required=True, help="Output file where data has to be stored.")

    args = parser.parse_args()

    create_profiling_data(
        shapes=parse_tensor_spec(args.shapes),
        value_ranges=parse_value_range(args.value_ranges),
        iterations=args.iterations,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
