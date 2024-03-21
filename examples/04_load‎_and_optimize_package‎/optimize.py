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
import logging

import model_navigator as nav

logger = logging.getLogger(__package__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
    )
    return parser.parse_args()


def main():
    """Load package, optimize and save new package"""
    args = parse_args()

    logger.info(f"Loading package from {args.input_path}.")
    """
    Load an existing package from the given input path.

    The load function automatically detects the used framework and package version
    based on the contents of the package."
    """
    package = nav.package.load(
        path=args.input_path,
    )

    logger.info("Optimize package.")
    """
    Re-run the conversions, correctness tests and profiling from existing package.
    Optimization creates new package.
    `package` API together with `save`, `load' and `optimize` methods helps with transferring models.
    It is possible to save the package, load it on different machine and run the optimization again.
    """
    package = nav.package.optimize(
        package=package,
        verbose=args.verbose,  # enable verbose logs
        debug=args.debug,  # enable debug logs
        defaults=(not args.no_defaults),  # use default optimization settings or settings from the package
    )

    logger.info(f"Save package to {args.output_path}.")
    """Save new package to the given output path."""
    nav.package.save(
        package=package,
        path=args.output_path,
        override=True,  # override the existing package if it exists.
    )

    logger.info("Package saved.")


if __name__ == "__main__":
    main()
