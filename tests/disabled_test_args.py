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
from typing import Any, NamedTuple

import argparse
import itertools
from argparse import _StoreAction

import pytest
from model_navigator.args import ArgParserGenerator


class ArgSpec(NamedTuple):
    name: str
    type: Any
    default: Any
    required: bool

    @staticmethod
    def from_store_action(action):
        return ArgSpec(
            name=action.option_strings[0], type=action.type, default=action.default, required=action.required
        )


def fn_with_no_args():
    pass


def fn1(param_foo: str, param_bar: int = 2):
    return {"param_foo": param_foo, "param_bar": param_bar}


class EmptyInitClass:
    def __init__(self):
        pass


class ClassWithArgsAndKwargs:
    def __init__(self, *args, **kwargs):
        pass


class ClassA:
    def __init__(self, param_foo: str, param_bar: int = 2):
        self.param_foo = param_foo
        self.param_bar = param_bar


class ClassB:
    def __init__(self, param_foo: str, param_bar: int = 2, **kwargs):
        self.param_foo = param_foo
        self.param_bar = param_bar
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "fn_or_clazz, expected_arg_specs",
    [
        (fn_with_no_args, []),
        (fn1, [ArgSpec("--param-foo", str, None, True), ArgSpec("--param-bar", int, 2, False)]),
        (EmptyInitClass, []),
        (ClassWithArgsAndKwargs, []),
        (ClassA, [ArgSpec("--param-foo", str, None, True), ArgSpec("--param-bar", int, 2, False)]),
        (ClassB, [ArgSpec("--param-foo", str, None, True), ArgSpec("--param-bar", int, 2, False)]),
        # add arguments with fn defined in python module
        (
            ClassA,
            [
                ArgSpec("--param-foo", str, None, True),
                ArgSpec("--param-bar", int, 2, False),
                ArgSpec("--data-dir", None, None, True),  # if not explicitly set, type is None
                ArgSpec("--batch-size", int, 1, False),
                ArgSpec("--images-num", int, None, False),
            ],
        ),
    ],
)
def test_arg_parser_generator_update(fn_or_clazz, expected_arg_specs):
    parser = argparse.ArgumentParser(description="short_description")
    ArgParserGenerator(fn_or_clazz).update_argparser(parser)
    arg_specs = [ArgSpec.from_store_action(action) for action in parser._actions if isinstance(action, _StoreAction)]
    for spec, expected_spec in itertools.zip_longest(arg_specs, expected_arg_specs):
        assert spec == expected_spec


@pytest.mark.parametrize(
    "fn, arguments_to_parse, expected_output",
    [
        (fn1, ["--param-foo", "arg1"], {"param_foo": "arg1", "param_bar": 2}),
    ],
)
def test_arg_parser_generator_fn_from_args(fn, arguments_to_parse, expected_output):
    parser = argparse.ArgumentParser(description="short_description")
    ArgParserGenerator(fn).update_argparser(parser)
    args = parser.parse_args(arguments_to_parse)
    output = ArgParserGenerator(fn).from_args(args)
    assert output == expected_output


@pytest.mark.parametrize(
    "clazz, arguments_to_parse, expected_output",
    [
        (ClassA, ["--param-foo", "arg1"], {"param_foo": "arg1", "param_bar": 2}),
        (
            ClassB,
            ["--param-foo", "arg1", "--data-dir", "my/path"],
            {
                "param_foo": "arg1",
                "param_bar": 2,
                "kwargs": {"data_dir": "my/path", "batch_size": 1, "images_num": None},
            },
        ),
    ],
)
def test_arg_parser_generator_class_from_args(clazz, arguments_to_parse, expected_output):
    parser = argparse.ArgumentParser(description="short_description")
    ArgParserGenerator(clazz).update_argparser(parser)
    args = parser.parse_args(arguments_to_parse)

    obj = ArgParserGenerator(clazz).from_args(args)
    for k, v in expected_output.items():
        assert getattr(obj, k) == v
