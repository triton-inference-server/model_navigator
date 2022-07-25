#!/usr/bin/env python3
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

import model_navigator as nav


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
    )
    parser.add_argument(
        "--output-path",
        type=str,
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
    )
    parser.add_argument(
        "--run-profiling",
        action="store_true",
    )

    return parser.parse_args()


# Re-run the conversions and profiling from existing package
def main():
    args = parse_args()

    pkg_desc = nav.load(
        path=args.input_path,
        retest_conversions=True,
        run_profiling=args.run_profiling,
    )

    output_path = args.output_path
    nav.save(pkg_desc, output_path, keep_workdir=args.keep_workdir, override=True)


if __name__ == "__main__":
    main()
