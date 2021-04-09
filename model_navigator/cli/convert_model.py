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

import logging
import shutil
import sys
from pathlib import Path

from model_navigator import Format
from model_navigator.cli import CLI
from model_navigator.log import log_dict, set_logger, set_tf_verbosity
from model_navigator.model import InputModel, guess_format
from model_navigator.model_navigator_exceptions import ModelNavigatorException
from model_navigator.optimizer.config import OptimizerConfig
from model_navigator.optimizer.pipelines import TransformersRegistry
from model_navigator.optimizer.runners import LocalTransformsExecutor

LOGGER = logging.getLogger("convert_model")


class ConversionError(ModelNavigatorException):
    pass


# TODO: could we always change precision?


def main():
    config = OptimizerConfig()
    cli = CLI(config)
    cli.add_argument("--output-path", required=True)
    args = cli.parse()

    set_logger(verbose=config.verbose)
    set_tf_verbosity(verbose=config.verbose)
    log_dict("convert_model.py config:", config.get_all_config())

    src_model = InputModel(
        name=config.model_name,
        path=Path(config.model_path),
    )
    target_format = (
        Format(config.target_format) if config.target_format else guess_format(model_path=Path(args.output_path))
    )
    config.target_format = target_format

    output_path = Path(args.output_path)
    registry = TransformersRegistry()
    transformers = registry.get(src_format=src_model.format, config=config)

    if not transformers:
        LOGGER.error(f"Could not find transformers for {src_model.format} to {target_format} conversion.")
        sys.exit(-1)
    elif len(transformers) > 1:
        LOGGER.warning(
            "There are more than one transformer obtained from registry for selected config. Executing first one."
        )
        transformers = transformers[:1]

    executor = LocalTransformsExecutor()
    result_models = executor.execute(transformers, src_model)
    if not result_models:
        LOGGER.error("Obtained no model")
        sys.exit(-1)

    result_model = result_models[0]
    shutil.copy(result_model.path, output_path)

    # copy also supplementary files - ex. model io annotation file
    # they have just changed suffix comparing to model path
    for supplementary_file in result_model.path.parent.glob(f"{result_model.path.stem}.*"):
        if supplementary_file == result_model.path:
            continue
        supplementary_file_output_path = output_path.parent / f"{output_path.stem}{supplementary_file.suffix}"
        shutil.copy(supplementary_file, supplementary_file_output_path)


if __name__ == "__main__":
    main()
