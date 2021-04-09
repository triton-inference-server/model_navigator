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

from model_navigator import Precision, Format
from model_navigator.cli import CLI
from model_navigator.config import ModelNavigatorConfig
from model_navigator.optimizer.config import OptimizerConfig


def test_config_concatenation():
    config = ModelNavigatorConfig().merge(OptimizerConfig())
    fields = config.get_config()
    assert "max_gpu_usage_mb" in fields
    assert "max_workspace_size" in fields


def test_config_set():
    config = OptimizerConfig()
    args = argparse.Namespace(model_name="foo", model_path="bar")
    config.set_config_values(args)
    assert config.target_precisions == [Precision.FP16, Precision.TF32]

    config.target_precisions = [Precision.FP32]
    assert config.target_precisions == [Precision.FP32]

    config.target_format = Format.ONNX
    assert config.target_format == Format.ONNX


def test_parsing_config_from_cli():
    config = OptimizerConfig()
    cli = CLI(config)

    argv = ["--model-name", "foo", "--model-path", "bar", "--onnx-opsets", "1", "2"]
    args = cli._parser.parse_args(argv)
    config.set_config_values(args)
    assert config.onnx_opsets == [1, 2]
    assert config.rtol == []
    assert config.atol == []
