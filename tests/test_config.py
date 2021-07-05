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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from model_navigator.model import Format
from model_navigator.triton.config import BackendAccelerator, TensorRTOptPrecision
from model_navigator.utils.config import BaseConfig, YamlConfigFile


@dataclass
class MyConfig(BaseConfig):
    config_a: int
    config_b: str
    config_c: Optional[bool] = None
    config_d: str = "foo"
    config_e: bool = False
    config_f: float = 1.0
    config_g: bool = True
    config_h: bool = True


@dataclass
class MyConfigOtherParams(BaseConfig):
    config2_a: int
    config2_b: bool = False


class MyConfigWithNested(BaseConfig):
    config: MyConfig


def test_config_from_dict_just_primitives():
    config = MyConfig.from_dict({"config_a": 1, "config_b": "foo"})  # just required parameters
    config = MyConfig.from_dict({"config_a": 1, "config_b": "foo", "config_c": False})  # with optional parameter
    assert config.config_c is False
    config = MyConfig.from_dict({"config_a": 1, "config_b": "foo", "config_z": 2})  # with additional parameters
    with pytest.raises(TypeError):
        config = MyConfig.from_dict({"config_a": 1})  # missing required parameter


def test_config_save_and_load():
    """Sava and load single config file"""
    config = MyConfig(config_a=8, config_b="foo")
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yml"
        with YamlConfigFile(config_path) as config_file:
            config_file.save_config(config)

        with YamlConfigFile(config_path) as config_file:
            loaded_config = config_file.load(MyConfig)

        assert config == loaded_config


def test_config_save_and_load_multiple_configs():
    """Sava and load 2 config files"""
    config1 = MyConfig(config_a=8, config_b="foo")
    config2 = MyConfigOtherParams(config2_a=8)
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yml"

        with YamlConfigFile(config_path) as config_file:
            config_file.save_config(config1)
            config_file.save_config(config2)

        with YamlConfigFile(config_path) as config_file:
            loaded_config1 = config_file.load(MyConfig)
            loaded_config2 = config_file.load(MyConfigOtherParams)

        assert config1 == loaded_config1
        assert config2 == loaded_config2


@dataclass
class MyConfigWithNestedAndEnumsListsAndDicts(BaseConfig):
    config_nested: MyConfig
    config_precision: TensorRTOptPrecision
    config_formats: List[Format]
    config_accelerators: Dict[str, BackendAccelerator]
    config_flag: bool


def test_config_save_and_load_more_complex():
    config = MyConfigWithNestedAndEnumsListsAndDicts(
        config_nested=MyConfig(config_a=8, config_b="foo"),
        config_precision=TensorRTOptPrecision.FP16,
        config_formats=[Format.ONNX, Format.TENSORRT],
        config_accelerators={"foo": BackendAccelerator.TRT, "bar": BackendAccelerator.AMP},
        config_flag=True,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yml"
        with YamlConfigFile(config_path) as config_file:
            config_file.save_config(config)

        with YamlConfigFile(config_path) as config_file:
            loaded_config = config_file.load(MyConfigWithNestedAndEnumsListsAndDicts)

    assert config == loaded_config


def test_config_conflict_detection():
    config1 = MyConfigWithNestedAndEnumsListsAndDicts(
        config_nested=MyConfig(config_a=8, config_b="foo"),
        config_precision=TensorRTOptPrecision.FP16,
        config_formats=[Format.ONNX, Format.TENSORRT],
        config_accelerators={"foo": BackendAccelerator.TRT, "bar": BackendAccelerator.AMP},
        config_flag=True,
    )
    config2 = MyConfig(config_a=8, config_b="bar")
    config3 = MyConfig(config_a=8, config_b="bar")
    config4 = MyConfig(config_a=18, config_b="bar")  # conflict on config_1
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yml"
        with YamlConfigFile(config_path) as config_file:
            config_file.save_config(config1)
            config_file.save_config(config2)  # should not raise because of MyConfig in config1 is nested
            config_file.save_config(
                config3
            )  # should not raise because of key value pairs between config2 and config3 are the same
            with pytest.raises(ValueError):
                config_file.save_config(config4)  # difference on config_a


def test_config_typing_checking():
    with pytest.raises(ValueError):
        MyConfig.from_dict({"config_a": "should_be_int", "config_b": "bar"})
