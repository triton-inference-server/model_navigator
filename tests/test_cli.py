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
import dataclasses
import functools
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

import click
import click.testing
import pytest

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.model import Format
from model_navigator.utils.cli import (
    CliSpec,
    common_options,
    is_dict_generic,
    is_list_generic,
    is_optional_generic,
    options_from_config,
)
from model_navigator.utils.config import BaseConfig, YamlConfigFile

# TODO: name conflicts check


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


def test_cli_with_just_simple_config(runner):
    """Try simple cmd invoke; also overwrite of default values"""

    @click.command()
    @options_from_config(MyConfig)
    def my_cmd_fun(**kwargs):
        config = MyConfig.from_dict(kwargs)
        print(config)

    expected_config_str = str(MyConfig(config_a=1, config_b="b", config_d="bar"))
    result = runner.invoke(my_cmd_fun, ["--config-a", "1", "--config-b", "b", "--config-d", "bar"])
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


def test_cli_with_just_config_with_bool_flags(runner):
    """Try cmd invoke with flags: toggle bool flags (false -> true and true  -> false),"""

    @click.command()
    @options_from_config(MyConfig)
    def my_cmd_fun(**kwargs):
        config = MyConfig.from_dict(kwargs)
        print(config)

    expected_config_str = str(
        MyConfig(config_a=1, config_b="b", config_c=True, config_d="bar", config_g=True, config_h=False)
    )
    result = runner.invoke(
        my_cmd_fun, ["--config-a", "1", "--config-b", "b", "--config-c", "--config-d", "bar", "--config-h"]
    )
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


@dataclass
class MyConfigWithEnums(BaseConfig):
    precision: TensorRTPrecision
    format: Format = Format.ONNX


def test_cli_for_config_with_enum(runner):
    """Check parsing enum values from CLI"""

    @click.command()
    @options_from_config(MyConfigWithEnums)
    def my_cmd_fun(**kwargs):
        config = MyConfigWithEnums.from_dict(kwargs)
        print(config)

    expected_config_str = str(MyConfigWithEnums(precision=TensorRTPrecision.FP16))
    result = runner.invoke(my_cmd_fun, ["--precision", "fp16"])
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0

    result = runner.invoke(my_cmd_fun, ["--precision", "foo"])
    assert "Invalid value for" in result.output
    assert result.exit_code == 2

    result = runner.invoke(my_cmd_fun, [])
    assert "Missing option" in result.output
    assert result.exit_code == 2

    # ensure enum value instead of enum as option default
    # FIXME
    # result = runner.invoke(my_cmd_fun, ["--help"])
    # assert "[default: onnx]" in result.output
    # assert result.exit_code == 0


@dataclass
class MyPrimitiveConfig(BaseConfig):
    config_i: int
    config_s: str
    config_f: float


class MyPrimitiveConfigCli:
    config_i = CliSpec(help="help for config_i")
    config_f = CliSpec(help="help for config_f")


def test_cli_help_from_cli_spec(runner):
    """Check adding help to click parser from spec"""

    @click.command()
    @options_from_config(MyPrimitiveConfig, MyPrimitiveConfigCli)
    def my_cmd_fun(**kwargs):
        config = MyPrimitiveConfig.from_dict(kwargs)
        print(config)

    result = runner.invoke(my_cmd_fun, ["--help"])
    assert not result.exception
    assert "help for config_i" in result.output
    assert "help for config_s" not in result.output
    assert "help for config_f" in result.output
    assert result.exit_code == 0

    result = runner.invoke(my_cmd_fun, [])
    assert "Missing option '--config-i'" in result.output  # why this message is missing?
    # assert "Missing option '--config-s'" in result.output
    # assert "Missing option '--config-f'" in result.output
    assert result.exception
    assert result.exit_code == 2


class MyPrimitiveConfigWithUnneededSpecCli:
    config_unneeded = CliSpec(help="help for unneeded param")


def test_cli_unneeded_cli_specs(runner):
    with pytest.raises(click.ClickException):

        @click.command()
        @options_from_config(MyPrimitiveConfig, MyPrimitiveConfigWithUnneededSpecCli)
        def my_cmd_fun(**kwargs):
            config = MyPrimitiveConfig.from_dict(kwargs)
            print(config)


@dataclass
class WithCountFlagConfig(BaseConfig):
    config_c: int = 0


class WithCountFlagConfigCli:
    config_c = CliSpec(help="help for config_c", count=True, param_decls=["-c", "--config-c"])


def test_cli_for_count_flag(runner):
    """Check presence of count flag"""

    @click.command()
    @options_from_config(WithCountFlagConfig, WithCountFlagConfigCli)
    def my_cmd_fun(**kwargs):
        config = WithCountFlagConfig.from_dict(kwargs)
        print(config)

    expected_config_str = str(WithCountFlagConfig(config_c=0))
    result = runner.invoke(my_cmd_fun, [])
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0

    expected_config_str = str(WithCountFlagConfig(config_c=1))
    result = runner.invoke(my_cmd_fun, ["--config-c"])
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0

    expected_config_str = str(WithCountFlagConfig(config_c=2))
    result = runner.invoke(my_cmd_fun, ["--config-c", "--config-c"])
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


def test_cli_params_aliases(runner):
    """Check presence of count flag"""

    @click.command()
    @options_from_config(WithCountFlagConfig, WithCountFlagConfigCli)
    def my_cmd_fun(**kwargs):
        config = WithCountFlagConfig.from_dict(kwargs)
        print(config)

    expected_config_str = str(WithCountFlagConfig(config_c=0))
    result = runner.invoke(my_cmd_fun, [])
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0

    expected_config_str = str(WithCountFlagConfig(config_c=1))
    result = runner.invoke(my_cmd_fun, ["-c"])
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0

    expected_config_str = str(WithCountFlagConfig(config_c=2))
    result = runner.invoke(my_cmd_fun, ["-cc"])
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


class WithCountFlagNotMatchingConfigCli:
    config_c = CliSpec(help="help for config_c", count=True, param_decls=["--not-matching-c", "-c"])


def test_cli_raise_exception_on_not_matching_name():
    """
    Check if ClickException is raised if explicitly provided first porameter doesn't match to dataclass parameter name
    """
    with pytest.raises(click.ClickException):

        @click.command()
        @options_from_config(WithCountFlagConfig, WithCountFlagNotMatchingConfigCli)
        def my_cmd_fun(**kwargs):
            config = WithCountFlagConfig.from_dict(kwargs)
            print(config)


def _parse_and_validate_nested_struct(ctx, param, value, cls):
    if isinstance(value, str):
        types = [str, int, str]
        parts = value.split(":")
        value = [cls_(part) for part, cls_ in zip(parts, types)]

    if isinstance(value, (list, tuple)) and not hasattr(value, "_fields"):
        return cls(*value)
    elif isinstance(value, dict):
        return cls(**value)
    else:
        return value


def _serialize_nested_struct(param, value):
    if dataclasses.is_dataclass(value):
        items = list(map(str, dataclasses.astuple(value)))
    else:
        items = list(value)
    return ":".join(items)


@dataclass
class NestedStructConfig(BaseConfig):
    part1: str
    part2: Optional[int] = 1
    part3: Optional[str] = "foo1"


@dataclass
class WithStructConfig(BaseConfig):
    config_nested_config2: NestedStructConfig
    config_nested_config: NestedStructConfig
    config_optional_nested_config2: Optional[NestedStructConfig] = None
    config_optional_nested_config: Optional[NestedStructConfig] = None


class WithStructConfigCli:
    config_nested_config2 = CliSpec(
        parse_and_verify_callback=functools.partial(_parse_and_validate_nested_struct, cls=NestedStructConfig),
    )
    config_nested_config = CliSpec(
        parse_and_verify_callback=functools.partial(_parse_and_validate_nested_struct, cls=NestedStructConfig),
    )
    config_optional_nested_config2 = CliSpec(
        parse_and_verify_callback=functools.partial(_parse_and_validate_nested_struct, cls=NestedStructConfig)
    )
    config_optional_nested_config = CliSpec(
        parse_and_verify_callback=functools.partial(_parse_and_validate_nested_struct, cls=NestedStructConfig)
    )


def test_cli_parse_params(runner):
    @click.command()
    @options_from_config(WithStructConfig, WithStructConfigCli)
    def my_cmd_fun(**kwargs):
        config = WithStructConfig.from_dict(kwargs)
        print(config)

    expected_config_str = str(
        WithStructConfig(
            config_nested_config2=NestedStructConfig("value1"), config_nested_config=NestedStructConfig("value1c")
        )
    )
    result = runner.invoke(my_cmd_fun, ["--config-nested-config2", "value1", "--config-nested-config", "value1c"])
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0

    expected_config_str = str(
        WithStructConfig(
            config_nested_config2=NestedStructConfig("value1", 10),
            config_nested_config=NestedStructConfig("value1c", 11),
        )
    )
    result = runner.invoke(my_cmd_fun, ["--config-nested-config2", "value1:10", "--config-nested-config", "value1c:11"])
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0

    expected_config_str = str(
        WithStructConfig(
            config_nested_config2=NestedStructConfig("value1", 10, "bar1"),
            config_nested_config=NestedStructConfig("value1c", 11, "bar1c"),
        )
    )
    result = runner.invoke(
        my_cmd_fun, ["--config-nested-config2", "value1:10:bar1", "--config-nested-config", "value1c:11:bar1c"]
    )
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0

    # expect click.BadParameter exception for passing non int param2
    result = runner.invoke(
        my_cmd_fun,
        ["--config-nested-config2", "value1:non_int_param", "--config-nested-config", "value1c:non_int_param"],
    )
    assert result.exception
    assert result.exit_code == 2


@dataclass
class MyConfigWithPrimitiveMultipleValues:
    config_a: int
    config_b: List[str]
    config_c: List[int] = field(default_factory=lambda: [1, 2, 3])
    config_d: bool = False


def test_config_from_cli_multiple_values():
    """Try cmd invoke with parameters of multiple values"""

    @click.command()
    @options_from_config(MyConfigWithPrimitiveMultipleValues)
    def my_cmd_fun(**kwargs):
        config = MyConfigWithPrimitiveMultipleValues(**kwargs)
        print(config)

    expected_config_str = str(
        MyConfigWithPrimitiveMultipleValues(config_a=1, config_b=["b1", "b2"], config_c=[2, 3, 4])
    )
    runner = click.testing.CliRunner()
    result = runner.invoke(
        my_cmd_fun,
        [
            "--config-a",
            "1",
            "--config-b",
            "b1",
            "b2",
            "--config-c",
            "2",
            "3",
            "4",
        ],
    )

    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


@dataclass
class MyConfigWithEnumMultipleValues:
    config_a: int
    config_b: List[Format]
    config_c: List[TensorRTPrecision] = field(default_factory=lambda: [TensorRTPrecision.FP16, TensorRTPrecision.TF32])
    config_d: bool = False


def test_config_from_cli_multiple_values_of_enums():
    """Try cmd invoke with parameters of multiple values including enums"""

    @click.command()
    @options_from_config(MyConfigWithEnumMultipleValues)
    def my_cmd_fun(**kwargs):
        config = MyConfigWithEnumMultipleValues(**kwargs)
        print(config)

    expected_config_str = str(
        MyConfigWithEnumMultipleValues(
            config_a=1,
            config_b=[Format.TENSORRT, Format.ONNX],
            config_c=[TensorRTPrecision.FP32, TensorRTPrecision.FP16],
        )
    )
    runner = click.testing.CliRunner()
    result = runner.invoke(
        my_cmd_fun,
        [
            "--config-a",
            "1",
            "--config-b",
            "trt",
            "onnx",
            "--config-c",
            "fp32",
            "fp16",
        ],
    )

    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


@dataclass
class MyConfigWithStructsMultipleValues:
    config_a: int
    config_b: List[NestedStructConfig]
    config_c: List[NestedStructConfig]
    config_d: List[NestedStructConfig] = field(
        default_factory=lambda: [
            NestedStructConfig("foo1c2", 121, "bar1c2"),
            NestedStructConfig("foo2c2", 122, "bar2c2"),
        ]
    )
    config_e: bool = False


class MyConfigWithStructsMultipleValuesCli:
    config_b = CliSpec(
        parse_and_verify_callback=functools.partial(_parse_and_validate_nested_struct, cls=NestedStructConfig)
    )
    config_c = CliSpec(
        parse_and_verify_callback=functools.partial(_parse_and_validate_nested_struct, cls=NestedStructConfig),
        serialize_default_callback=_serialize_nested_struct,
    )
    config_d = CliSpec(
        parse_and_verify_callback=functools.partial(_parse_and_validate_nested_struct, cls=NestedStructConfig),
        serialize_default_callback=_serialize_nested_struct,
    )


def test_config_from_cli_multiple_values_of_structs():
    """Try cmd invoke with parameters of multiple values including structs"""

    @click.command()
    @options_from_config(MyConfigWithStructsMultipleValues, MyConfigWithStructsMultipleValuesCli)
    def my_cmd_fun(**kwargs):
        config = MyConfigWithStructsMultipleValues(**kwargs)
        print(config)

    expected_config_str = str(
        MyConfigWithStructsMultipleValues(
            config_a=1,
            config_b=[NestedStructConfig("foo1", 1, "bar1"), NestedStructConfig("foo2", 2, "bar2")],
            config_c=[NestedStructConfig("foo1c1", 111, "bar1c1"), NestedStructConfig("foo2c1", 112, "bar2c1")],
        )
    )
    runner = click.testing.CliRunner()
    result = runner.invoke(
        my_cmd_fun,
        [
            "--config-a",
            "1",
            "--config-b",
            "foo1:1:bar1",
            "foo2:2:bar2",
            "--config-c",
            "foo1c1:111:bar1c1",
            "foo2c1:112:bar2c1",
        ],
    )

    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


def test_cli_order_match_fields_order(runner):
    @click.command()
    @options_from_config(MyConfig)
    def my_cmd_fun(**kwargs):
        config = MyConfig.from_dict(kwargs)
        print(config)

    result = runner.invoke(my_cmd_fun, ["--help"])
    assert not result.exception
    lines = result.output.splitlines()
    start_idx = 3
    for idx, field_ in enumerate(fields(MyConfig)):
        option_name = f"--{field_.name.replace('_', '-')}"
        assert option_name in lines[start_idx + idx]
    assert result.exit_code == 0


def test_is_optional_generic():
    assert is_optional_generic(Optional[int])
    assert is_optional_generic(Optional[List[str]])
    assert is_optional_generic(Optional[Any])
    assert not is_optional_generic(List)
    assert not is_optional_generic(List[str])
    assert not is_optional_generic(Dict[str, str])


def test_is_list_generic():
    assert not is_list_generic(Optional[int])
    assert is_list_generic(Optional[List[str]])
    assert is_list_generic(Optional[List[Any]])
    assert not is_list_generic(Optional[Any])
    assert is_list_generic(List)
    assert is_list_generic(List[str])
    assert not is_list_generic(Dict[str, str])


def test_is_dict_generic():
    assert not is_dict_generic(Optional[int])
    assert not is_dict_generic(Optional[List[str]])
    assert not is_dict_generic(Optional[List[Any]])
    assert not is_dict_generic(Optional[Any])
    assert not is_dict_generic(List)
    assert not is_dict_generic(List[str])
    assert is_dict_generic(Dict)
    assert is_dict_generic(Dict[str, str])
    assert is_dict_generic(Dict[str, Any])
    assert is_dict_generic(Optional[Dict[str, str]])
    assert is_dict_generic(Optional[Dict[str, Any]])
    assert is_dict_generic(Optional[Dict])


def test_load_from_config_file_primitives(runner):
    @click.command()
    @common_options
    @options_from_config(MyConfig)
    def my_cmd_fun(**kwargs):
        config = MyConfig.from_dict(kwargs)
        print(config)

    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        config = MyConfig(config_a=1, config_b="b", config_d="bar")
        with YamlConfigFile(config_path) as config_file:
            config_file.save_config(config)

        expected_config_str = str(config)
        result = runner.invoke(my_cmd_fun, ["--config-path", config_path])
        assert not result.exception
        assert result.output.splitlines() == [expected_config_str]
        assert result.exit_code == 0


# Tests of loading config from file
def _save_config_file_and_invoke_cmd(config, cmd_fn, runner):
    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        with YamlConfigFile(config_path) as config_file:
            config_file.save_config(config)

        with config_path.open("r") as config_file:
            payload = config_file.read()
            print(payload)

        return runner.invoke(cmd_fn, ["--config-path", config_path])


def test_cli_with_load_from_config_file_simple(runner):
    @click.command()
    @common_options
    @options_from_config(MyConfig)
    def my_cmd_fun(**kwargs):
        config = MyConfig.from_dict(kwargs)
        print(config)

    config = MyConfig(config_a=1, config_b="b", config_d="bar")

    result = _save_config_file_and_invoke_cmd(config, my_cmd_fun, runner)

    expected_config_str = str(config)
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


def test_cli_with_load_from_config_with_enum(runner):
    """Check parsing enum values from CLI"""

    @click.command()
    @common_options
    @options_from_config(MyConfigWithEnums)
    def my_cmd_fun(**kwargs):
        config = MyConfigWithEnums.from_dict(kwargs)
        print(config)

    config = MyConfigWithEnums(precision=TensorRTPrecision.FP16)
    expected_config_str = str(config)

    result = _save_config_file_and_invoke_cmd(config, my_cmd_fun, runner)

    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


def test_cli_with_load_from_config_parse_params(runner):
    @click.command()
    @common_options
    @options_from_config(WithStructConfig, WithStructConfigCli)
    def my_cmd_fun(**kwargs):
        config = WithStructConfig.from_dict(kwargs)
        print(config)

    config = WithStructConfig(
        config_nested_config2=NestedStructConfig("value1", 10, "bar1"),
        config_nested_config=NestedStructConfig("value1c", 11),
    )
    expected_config_str = str(config)
    result = _save_config_file_and_invoke_cmd(config, my_cmd_fun, runner)

    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


@dataclass
class WithListOfPrimitives(BaseConfig):
    list_of_ints: List[int]
    list_of_floats: List[float]
    list_of_str: List[str]


def test_cli_with_list_of_primitives_1(runner):
    @click.command()
    @options_from_config(WithListOfPrimitives)
    def my_cmd_fun(**kwargs):
        config = WithListOfPrimitives.from_dict(kwargs)
        print(config)

    config = WithListOfPrimitives([0, 1, 2], [0.0, 1.0, 2.0], ["a", "b", "c"])
    expected_config_str = str(config)
    result = runner.invoke(
        my_cmd_fun,
        [
            "--list-of-ints",
            "0",
            "1",
            "2",
            "--list-of-floats",
            "0.",
            "1.",
            "2.",
            "--list-of-str",
            "a",
            "b",
            "c",
        ],
    )
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


@dataclass
class WithOptionalListOfPrimitives(BaseConfig):
    list_of_ints: Optional[List[int]] = None
    list_of_floats: Optional[List[float]] = None
    list_of_str: Optional[List[str]] = None


def test_cli_with_list_of_primitives_2(runner):
    @click.command()
    @options_from_config(WithOptionalListOfPrimitives)
    def my_cmd_fun(**kwargs):
        config = WithOptionalListOfPrimitives.from_dict(kwargs)
        print(config)

    config = WithOptionalListOfPrimitives(list_of_ints=[0, 1, 2])
    expected_config_str = str(config)
    result = runner.invoke(
        my_cmd_fun,
        [
            "--list-of-ints",
            "0",
            "1",
            "2",
        ],
    )
    assert not result.exception
    assert result.output.splitlines() == [expected_config_str]
    assert result.exit_code == 0


def test_cli_return_error_code_on_missing_conversion_outputs():
    with TemporaryDirectory() as temp_dir:
        completed_process_result = subprocess.run(
            [
                "model-navigator",
                "convert",
                "--model-name",
                "MyModel",
                "--model-path",
                "tests/files/models/identity.savedmodel",
                "--target-formats",
                "torchscript",
                "--output-path",
                (Path(temp_dir) / "model.onnx").as_posix(),
                "--launch-mode",
                "local",
                "--override-workspace",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print("output", completed_process_result.stdout.decode("utf-8"))
        assert completed_process_result.returncode != 0
