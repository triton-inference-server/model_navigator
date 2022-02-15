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
import tempfile
from pathlib import Path

import click
import numpy as np

from model_navigator.cli.spec import ComparatorConfigCli, ModelSignatureConfigCli
from model_navigator.converter.config import ComparatorConfig, DatasetProfileConfig
from model_navigator.model import ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.utils.cli import common_options, options_from_config
from model_navigator.utils.config import YamlConfigFile

MODEL_SIGNATURE_CONFIG1 = ModelSignatureConfig(
    inputs={
        "input1": TensorSpec("tensor1", (-1, 3, -1, -1), dtype=np.dtype("float32")),
        "input2": TensorSpec("tensor2:0", (-1, 1), dtype=np.dtype("int64")),
    },
    outputs={"output": TensorSpec("output_tensor:0", (-1, 1000), dtype=np.dtype("float32"))},
)
MODEL_SIGNATURE_CONFIG1_CMDS = [
    "--inputs",
    "input1=tensor1:-1,3,-1,-1:float32",
    "input2=tensor2:0:-1,1:int64",
    "--outputs",
    "output=output_tensor:0:-1,1000:float32",
]

DATASET_PROFILE_CONFIG1 = DatasetProfileConfig(
    min_shapes={"input1": (1, 3, 16, 16), "input2": (1, 1)},
    max_shapes={"input1": (256, 3, 320, 320), "input2": (256, 1)},
    value_ranges={"input1": (0.0, 1.0), "input2": (0, 128)},
)

DATASET_PROFILE_CONFIG1_CMDS = [
    "--min-shapes",
    "input1=1,3,16,16",
    "input2=1,1",
    "--max-shapes",
    "input1=256,3,320,320",
    "input2=256,1",
    "--value-ranges",
    "input1=0.,1.",
    "input2=0,128",
]

COMPARATOR_CONFIG1 = ComparatorConfig(
    atol={"": 1e-5, "output": 0.005},
)
COMPARATOR_CONFIG1_CMDS = ["--atol", "=1e-5", "output=0.005"]


def test_cli_with_model_signature_from_cli(runner):
    @click.command()
    @options_from_config(ModelSignatureConfig, ModelSignatureConfigCli)
    def my_cmd_fun(**kwargs):
        config = ModelSignatureConfig.from_dict(kwargs)
        print(config)

    expected_config_str = str(MODEL_SIGNATURE_CONFIG1)

    result = runner.invoke(
        my_cmd_fun,
        MODEL_SIGNATURE_CONFIG1_CMDS,
    )
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


def test_cli_with_model_signature_from_config(runner):
    """Check parsing inputs and outputs from config"""

    @click.command()
    @common_options
    @options_from_config(ModelSignatureConfig, ModelSignatureConfigCli)
    def my_cmd_fun(**kwargs):
        config = ModelSignatureConfig.from_dict(kwargs)
        print(config)

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        with YamlConfigFile(config_path) as config_file:
            config_file.save_config(MODEL_SIGNATURE_CONFIG1)

        expected_config_str = str(MODEL_SIGNATURE_CONFIG1)
        result = runner.invoke(my_cmd_fun, ["--config-path", config_path.resolve().as_posix()])
        assert result.output.splitlines() == [expected_config_str]
        assert not result.exception
        assert result.exit_code == 0


def test_cli_with_comparator_config_from_cli(runner):
    @click.command()
    @options_from_config(ComparatorConfig, ComparatorConfigCli)
    def my_cmd_fun(**kwargs):
        config = ComparatorConfig.from_dict(kwargs)
        print(config)

    expected_config_str = str(COMPARATOR_CONFIG1)

    result = runner.invoke(
        my_cmd_fun,
        COMPARATOR_CONFIG1_CMDS,
    )
    assert result.output.splitlines() == [expected_config_str]
    assert not result.exception
    assert result.exit_code == 0


def test_cli_with_comparator_config_from_config(runner):
    """Check parsing inputs and outputs from config"""

    @click.command()
    @common_options
    @options_from_config(ComparatorConfig, ComparatorConfigCli)
    def my_cmd_fun(**kwargs):
        config = ComparatorConfig.from_dict(kwargs)
        print(config)

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        with YamlConfigFile(config_path) as config_file:
            config_file.save_config(COMPARATOR_CONFIG1)

        expected_config_str = str(COMPARATOR_CONFIG1)
        result = runner.invoke(my_cmd_fun, ["--config-path", config_path.resolve().as_posix()])
        assert result.output.splitlines() == [expected_config_str]
        assert not result.exception
        assert result.exit_code == 0
