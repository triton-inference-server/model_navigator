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
import dataclasses
import functools
import logging
from dataclasses import MISSING, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import click
import yaml
from click.types import UNPROCESSED

from model_navigator.core import DEFAULT_CONTAINER_VERSION
from model_navigator.utils import nav_package
from model_navigator.utils.config import YamlConfigFile
from model_navigator.utils.workspace import DEFAULT_WORKSPACE_PATH, Workspace

LOGGER = logging.getLogger(__name__)


@dataclass
class CliSpec:
    help: str = ""
    count: bool = False
    multiple: bool = False
    param_decls: Optional[List[str]] = None
    parse_and_verify_callback: Optional[Callable] = None
    serialize_default_callback: Optional[Callable] = None
    default_factory: Optional[Callable] = None


class OptionNargs(click.Option):
    """
    modified version of:
    https://github.com/aws/aws-sam-cli/blob/develop/samcli/commands/_utils/custom_options/option_nargs.py
    which is copy of https://stackoverflow.com/a/48394004

    A custom option class that allows parsing for multiple arguments
    for an option, when the number of arguments for an option are unknown.
    """

    def __init__(self, *args, **kwargs):
        self.nargs = kwargs.pop("nargs", -1)
        super().__init__(*args, **kwargs)
        self._nargs_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # look ahead into arguments till we reach the next option.
            # the next option starts with a prefix which is either '-' or '--'
            next_option = False
            value = [value]

            parser_ = self._nargs_parser
            while state.rargs and not next_option:
                for prefix in parser_.prefixes:
                    if state.rargs[0].startswith(prefix):
                        next_option = True
                if not next_option:
                    value.append(state.rargs.pop(0))

            value = list(value)

            # call the actual OptionParser.process (below modified version)
            if parser_.action == "store":
                state.opts[parser_.dest] = value
            elif parser_.action == "store_const":
                state.opts[parser_.dest] = parser_.const
            elif parser_.action == "append":
                dest = state.opts.setdefault(parser_.dest, [])
                if isinstance(value, (list, tuple)):  # added this check in order to extend list
                    dest.extend(value)
                else:
                    dest.append(value)
            elif parser_.action == "append_const":
                dest = state.opts.setdefault(parser_.dest, [])
                dest.append(parser_.const)
                if isinstance(parser_.const, (list, tuple)):  # added this check in order to extend list
                    dest.extend(parser_.const)
                else:
                    dest.append(value)
            elif parser_.action == "count":
                state.opts[parser_.dest] = state.opts.get(parser_.dest, 0) + 1
            else:
                raise ValueError(f"unknown action '{parser_.action}'")
            state.order.append(parser_.obj)

        # Add current option to Parser by calling add_to_parser on the super class.
        super().add_to_parser(parser, ctx)
        for name in self.opts:
            # Get OptionParser object for current option
            option_parser = getattr(parser, "_long_opt").get(name) or getattr(parser, "_short_opt").get(name)
            if option_parser:
                # Monkey patch `process` method for click.parser.Option class.
                # This allows for setting multiple parsed values into current option arguments
                self._nargs_parser = option_parser
                option_parser.process = parser_process
                break

    def type_cast_value(self, ctx, value):
        """
        Modified copy of code from Parameter.type_cast_value

        Given a value this runs it properly through the type system.
        This automatically handles things like `nargs` and `multiple` as
        well as composite types.
        """
        if self.type.is_composite:
            if self.nargs <= 1:
                raise TypeError(
                    "Attempted to invoke composite type but nargs has"
                    " been set to {}. This is not supported; nargs"
                    " needs to be set to a fixed value > 1.".format(self.nargs)
                )
            if self.multiple:
                container_type = type(value)  # added this line to ensure same type
                return container_type(self.type(x or (), self, ctx) for x in value or ())
            return self.type(value or (), self, ctx)

        def _convert(value_, level):
            if value_ is None:
                return value_
            if level == 0:
                return self.type(value_, self, ctx)
            _container_type = type(value_)  # added this line to ensure same type (list/tuple)
            return _container_type(_convert(x, level - 1) for x in value_ or ())

        return _convert(value, (self.nargs != 1) + bool(self.multiple))


def _extract_enum_values_if_present(param, value):
    if isinstance(value, Enum):
        value = value.value
    elif isinstance(value, (list, tuple)) and isinstance(value[0], Enum):
        container_type = type(value)
        value = container_type([item.value for item in value])
    elif isinstance(value, dict) and isinstance(list(value.values())[0], Enum):
        value = {k: v.value for k, v in value.items()}
    return value


# TODO: make replace Command.make_parser function thus do not need to monkey patch OptionParser


def is_optional_generic(type_):
    from typing_inspect import is_optional_type

    return is_optional_type(type_)


def is_list_generic(type_):
    from typing_inspect import get_args, get_origin, is_generic_type

    is_optional = is_optional_generic(type_)
    if is_optional:
        type_, _ = get_args(type_, evaluate=True)
    return is_generic_type(type_) and get_origin(type_) in [list, List]


def is_dict_generic(type_):
    from typing_inspect import get_args, get_origin, is_generic_type

    is_optional = is_optional_generic(type_)
    if is_optional:
        type_, _ = get_args(type_, evaluate=True)
    return is_generic_type(type_) and get_origin(type_) in [dict, Dict]


def is_namedtuple(type_):
    from typing_inspect import get_args, get_origin, is_generic_type, is_union_type

    is_optional = is_optional_generic(type_)
    if is_optional:
        type_, _ = get_args(type_, evaluate=True)

    is_generic_or_union = is_generic_type(type_) or is_union_type(type_)
    generic_tuple = is_generic_type(type_) and get_origin(type_) in [tuple, Tuple]
    is_namedtuple = generic_tuple or (
        not is_generic_or_union and issubclass(type_, tuple) and hasattr(type_, "_fields")
    )

    return is_namedtuple


def _parse_and_verify_callback_wrapper(parse_and_verify_callback: Optional[Callable] = None):
    @functools.wraps(parse_and_verify_callback)
    def _wrapper(ctx, param, value):
        if value is not None and parse_and_verify_callback is not None:
            # if scalar parameter and callback is provided - just pass it to
            try:
                if not param.multiple:
                    # initial value can be serialized cli string, dict from config file, tuple/list of struct attributes
                    value = parse_and_verify_callback(ctx, param, value)
                else:
                    container_type = type(value)
                    value = container_type([parse_and_verify_callback(ctx, param, v) for v in value])
            except ValueError as e:
                raise click.BadParameter(
                    f"{param} should be <part1:str>[:<part2:int>[:<part3:str>]] but got {value}; error details: {e}"
                )
        return value

    return _wrapper


def _serialize_default_callback_wrapper(
    serialize_default_callback: Optional[Callable] = None, param_multiple: bool = False
):
    @functools.wraps(serialize_default_callback)
    def _wrapper(param_name, value):
        if value is not None and serialize_default_callback is not None:
            if not param_multiple:
                value = serialize_default_callback(param_name, value)
            else:
                container_type = type(value)
                value = container_type([serialize_default_callback(param_name, item) for item in value])
        return value

    return _wrapper


def options_from_config(config_dataclass, cli_specs=None):  # noqa: C901
    """Converts dataclass into sequence of click.Options. Uses CliSpec class for additional CLI specific metadata."""

    if cli_specs:
        # verify if all cli specs are needed
        cli_spec_fields_set = {field_name for field_name in cli_specs.__dict__ if not field_name.startswith("__")}
        dataclass_fields_set = {f.name for f in fields(config_dataclass)}
        additional_fields = cli_spec_fields_set - dataclass_fields_set

        if additional_fields:
            raise click.ClickException(
                f"{cli_specs} class contains parameters: {', '.join(additional_fields)} "
                f"which doesn't match with {config_dataclass} config"
            )

    def _from_field_to_option(dataclass_field):
        cli_spec = getattr(cli_specs, dataclass_field.name, None) or CliSpec()

        # check presence of full_option matching field name
        hyphened_name = dataclass_field.name.replace("_", "-")
        param_decls = cli_spec.param_decls if cli_spec.param_decls is not None else [f"--{hyphened_name}"]
        full_param_decls = [pd for pd in param_decls if pd.startswith("--")]
        possible_name = full_param_decls[0][2:].replace("-", "_")
        if possible_name != dataclass_field.name:
            raise click.ClickException(
                f"Provided parameters names: {param_decls} doesn't match field name: {dataclass_field.name}"
            )

        # default option kwargs
        cls = click.Option
        if not cli_spec.multiple:
            parse_and_verify_callback = _parse_and_verify_callback_wrapper(cli_spec.parse_and_verify_callback)
        else:
            # don't do any magic, if the argument is explicitly specified as multiple
            parse_and_verify_callback = cli_spec.parse_and_verify_callback
        nargs = None
        flag_value = None

        is_optional = is_optional_generic(dataclass_field.type)
        if is_optional:
            field_type = dataclass_field.type.__args__[0]  # Optional[cls] will be changed into Union[cls, None]
        else:
            field_type = dataclass_field.type

        is_list = is_list_generic(field_type)
        is_dict = is_dict_generic(field_type)
        is_dataclass = dataclasses.is_dataclass(field_type)
        is_required = (
            dataclass_field.default == MISSING
            and dataclass_field.default_factory == MISSING
            and not cli_spec.default_factory
            and not is_optional
        )

        # TODO: fix this - obtain args for each generic list/optional and then decide on type based on it
        # obtain type, and option class
        if is_list:
            type_ = field_type.__args__[0]  # List[cls] -> cls
            cls = OptionNargs
            nargs = "*"
        elif is_dict:
            assert (
                cli_spec.parse_and_verify_callback
            ), f"{dataclass_field.name} is dict type please provide parse_and_verify_callback"
            cls = OptionNargs
            nargs = "*"
        elif is_dataclass:
            assert (
                cli_spec.parse_and_verify_callback
            ), f"{dataclass_field.name} is dataclass type please provide parse_and_verify_callback"
        else:
            type_ = field_type

        if cli_spec.parse_and_verify_callback:
            type_ = UNPROCESSED

        # obtain default
        default = None
        if cli_spec.default_factory:
            default = cli_spec.default_factory()
        elif dataclass_field.default_factory != MISSING:
            default = dataclass_field.default_factory()
        elif dataclass_field.default != MISSING:
            default = dataclass_field.default

        if default:
            is_enum = (
                isinstance(default, Enum)
                or (isinstance(default, (list, tuple)) and isinstance(default[0], Enum))
                or (isinstance(default, dict) and isinstance(list(default.values())[0], Enum))
            )

            if default and cli_spec.parse_and_verify_callback and not cli_spec.serialize_default_callback:
                raise RuntimeError(
                    f"For {config_dataclass.__name__}.{dataclass_field.name} default ({default}) "
                    f"and parse_and_verify_callback is provided but serialize_default_callback is missing"
                )

            serialize_default_callback = cli_spec.serialize_default_callback
            if is_enum and serialize_default_callback is None:
                serialize_default_callback = _extract_enum_values_if_present

            serialize_default_callback = _serialize_default_callback_wrapper(serialize_default_callback, is_list)
            if default and serialize_default_callback:
                default = serialize_default_callback(dataclass_field.name, default)

        # and flag_value
        is_bool_flag = type_ == bool
        has_default = default is not None
        if is_bool_flag and has_default:
            flag_value = not default

        option_kwargs = {
            "default": default,
            "show_default": True,
            "cls": cls,
            "is_flag": is_bool_flag,
            "flag_value": flag_value,
            "count": cli_spec.count,
            "multiple": is_list or cli_spec.multiple,
            "help": cli_spec.help,
            "type": type_,
            "required": is_required,
            "callback": parse_and_verify_callback,
            "nargs": nargs,
        }

        return click.option(*param_decls, **option_kwargs)

    def wrapper_fn(cmd):
        options = [_from_field_to_option(field_) for field_ in fields(config_dataclass)]

        options.reverse()
        for option in options:
            option(cmd)

        return cmd

    return wrapper_fn


def common_options(f):
    from model_navigator.kubernetes.triton import TritonServer

    def _load_config_from_file(ctx, param, value):
        """Set other CLI options defaults based on parameters from config file"""
        if not value:
            return value

        config_path = Path(value)
        ctx.default_map = ctx.default_map or {}
        with YamlConfigFile(config_path=config_path) as config_file:
            ctx.default_map.update(config_file.config_dict)

        return config_path

    def _load_config_from_nav_package(ctx, param, value):
        """Set other CLI options defaults based on parameters from nav package"""
        if not value:
            return

        package_path = Path(value)
        package = nav_package.from_path(package_path)
        with package.open("status.yaml") as f:
            status = yaml.load(f, Loader=yaml.SafeLoader)

        ctx.default_map = ctx.default_map or {}

        model = nav_package.select_input_format(status["model_status"])
        LOGGER.info("Selected model %s as input", model)
        config_path = Path(model["path"]).parent / "config.yaml"

        with package.open(config_path) as config_file:
            config_dict = yaml.safe_load(config_file)
            model_path = config_dict["model_path"]
            ctx.default_map.update(config_dict)
            # we need to fix the relative path
            ctx.default_map.update({"model_path": (package, model_path)})

        return package

    arguments = [
        # package and config-path should be read before all the options,
        # as callbacks read config files into ctx.default_map
        click.argument(
            "package",
            type=click.Path(dir_okay=True, file_okay=True, exists=True, resolve_path=True),
            required=False,
            callback=_load_config_from_nav_package,
        ),
    ]

    # TODO: add verification if --config-path is first option wrapping command
    options = [
        # package and config-path should be read before all other options,
        # as callbacks read config files into ctx.default_map
        click.option(
            "--config-path",
            help=(
                "Path to the configuration file containing default parameter values to use. "
                "\nFor more information about configuration files, refer to: "
                "\nhttps://github.com/triton-inference-server/model_navigator/blob/main/docs/run.md"
            ),
            type=click.Path(dir_okay=False, exists=True),
            show_default=True,
            required=False,
            callback=_load_config_from_file,
        ),
        click.option(
            "--workspace-path",
            help="Path to the output workspace directory.",
            type=click.Path(file_okay=False, writable=True),
            default=DEFAULT_WORKSPACE_PATH,
            show_default=True,
            required=False,
        ),
        click.option(
            "-o",
            "--output-package",
            help="Path to the output package.",
            type=click.Path(file_okay=True, dir_okay=False, writable=True),
            default=None,
            show_default=True,
            required=False,
        ),
        click.option(
            "--override-workspace",
            help="Clean workspace directory before command execution.",
            is_flag=True,
            default=False,
        ),
        click.option(
            "--container-version",
            help=(
                "NVIDIA framework and Triton container version to use. For details refer to"
                "\nhttps://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html and"
                "\nhttps://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html for details)."
            ),
            default=DEFAULT_CONTAINER_VERSION,
        ),
        click.option(
            "--framework-docker-image",
            help=(
                "Custom framework docker image to use. If not provided "
                "\nnvcr.io/nvidia/<framework>:<container_version>-<framework_and_python_version> will be used"
            ),
            type=str,
            required=False,
        ),
        click.option(
            "--triton-docker-image",
            help=(
                "Custom Triton Inference Server docker image to use. "
                f"\nIf not provided {TritonServer.image}:<container_version>-{TritonServer.tag} will be used"
            ),
            type=str,
            required=False,
        ),
        click.option(
            "--gpus",
            help=(
                "List of GPU UUIDs or Device IDs to be used for the conversion and/or profiling."
                "\nAll values have to be provided in the same format. "
                "\nUse 'all' to profile all the GPUs visible by CUDA."
            ),
            default=["all"],
            show_default=True,
            multiple=True,
        ),
        click.option(
            "-v",
            "--verbose",
            help="Provide verbose logs.",
            default=False,
            type=bool,
            is_flag=True,
        ),
        click.option(
            "--random-seed",
            help="Seed to use for random number generation.",
            default=0,
            type=int,
        ),
    ]

    options.reverse()
    for option in options:
        f = option(f)

    for argument in reversed(arguments):
        f = argument(f)

    return f


def clean_workspace_if_needed(workspace: Workspace, override_workspace: bool):
    if not workspace.empty():
        if override_workspace:
            workspace.clean()
        else:
            LOGGER.warning(
                f"Workspace {workspace.path} is not empty. "
                f"Use --override-workspace flag to force workspace directory cleaning before command execution."
            )
            raise click.Abort()
