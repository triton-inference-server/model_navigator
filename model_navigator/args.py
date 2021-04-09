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
from typing import Any, Callable, Dict, Optional, Union

import argparse
import inspect
import logging

LOGGER = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def filter_fn_args(args: Union[dict, argparse.Namespace], fn: Callable) -> dict:
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    signature = inspect.signature(fn)
    parameters_names = list(signature.parameters)
    args = {k: v for k, v in args.items() if k in parameters_names}
    return args


def add_args_for_fn_signature(parser: argparse._ActionsContainer, fn: Callable) -> argparse._ActionsContainer:
    parser.conflict_handler = "resolve"
    signature = inspect.signature(fn)
    for parameter in signature.parameters.values():
        if parameter.name in ["self", "args", "kwargs"]:
            continue
        argument_kwargs = {}
        if parameter.annotation != inspect.Parameter.empty:
            if parameter.annotation == bool:
                argument_kwargs["type"] = str2bool
                argument_kwargs["choices"] = [0, 1]
            else:
                argument_kwargs["type"] = parameter.annotation
        if parameter.default != inspect.Parameter.empty:
            if parameter.annotation == bool:
                argument_kwargs["default"] = str2bool(parameter.default)
            else:
                argument_kwargs["default"] = parameter.default
        else:
            argument_kwargs["required"] = True
        name = parameter.name.replace("_", "-")
        LOGGER.debug(f"Adding argument {name} with {argument_kwargs}")
        parser.add_argument(f"--{name}", **argument_kwargs)
    return parser


class ArgParserGenerator:
    def __init__(self, fn: Callable, argparse_update_fn: Optional[Callable[[argparse._ActionsContainer], None]] = None):
        if not inspect.isfunction(fn):
            raise ValueError(f"fn argument should be function; got {fn.__qualname__}")

        self._fn = fn
        self._argparse_update_fn = argparse_update_fn

    def update_argparser(self, parser: argparse.ArgumentParser):
        """
        Update provided ArgumentParser with arguments matching wrapped function signature.

        Arguments set is obtained with provided argparse_update_fn or if that function is missing,
        they are generated based on parameters of function signature.

        :param parser: parser to update
        :return: None
        """
        name = self._fn.__name__
        group_parser = parser.add_argument_group(name)
        if self._argparse_update_fn:
            self._argparse_update_fn(group_parser)
        else:
            add_args_for_fn_signature(group_parser, fn=self._fn)

    def get_args(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Get arguments required to call wrapped function
        :param args: all arguments obtained from ArgumentParser
        :return: kwargs matching wrapped function signature
        """

        if self._argparse_update_fn:
            filtered_args = self._filter_based_on_argparser_update_fn(args)
        else:
            filtered_args = filter_fn_args(args, fn=self._fn)

        return filtered_args

    def _filter_based_on_argparser_update_fn(self, args):
        tmp_parser = argparse.ArgumentParser()
        self._argparse_update_fn(tmp_parser)
        custom_names = [
            action.dest.replace("-", "_")
            for action in tmp_parser._actions
            if not isinstance(action, argparse._HelpAction)
        ]
        filtered_args = {n: getattr(args, n) for n in custom_names}
        return filtered_args

    def from_args(self, args: Union[argparse.Namespace, Dict]):
        kwargs = self.get_args(args)
        LOGGER.info(f"Initializing {self._fn.__name__}({kwargs})")
        return self._fn(**kwargs)
