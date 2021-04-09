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
from typing import Optional, Union

from model_navigator import Format, Precision
from model_navigator.config import ModelNavigatorBaseConfig
from model_navigator.config.config_field import ConfigField
from model_navigator.config.config_list_numeric import ConfigListNumeric
from model_navigator.config.config_list_string import ConfigListString
from model_navigator.config.config_primitive import ConfigPrimitive
from model_navigator.tensor import TensorSpec

LOGGER = logging.getLogger(__name__)

TRITON_SUPPORTED_FORMATS = [
    Format.TS_TRACE,
    Format.TS_SCRIPT,
    Format.TF_SAVEDMODEL,
    Format.ONNX,
    Format.TRT,
]


def parse_tensor_spec(shapes):
    return [TensorSpec.from_command_line(s) for s in shapes]


def parse_precisions(precisions):
    return [p if isinstance(p, Precision) else Precision(p) for p in precisions]


def parse_format(format_: Optional[Union[str, Format]]):
    if format_ is None or isinstance(format_, Format):
        return format_
    return Format(format_)


def parse_value_range(value_ranges):
    results = []
    for value_range in value_ranges:
        name, range_str = value_range.split(":")
        min_value, max_value = range_str.split(",")

        has_dot = "." in min_value or "." in max_value
        min_value, max_value = float(min_value), float(max_value)
        if min_value.is_integer() and max_value.is_integer() and not has_dot:
            min_value, max_value = int(min_value), int(max_value)

        results.append((name, (min_value, max_value)))
    return results


def parse_error_thresholds(thresholds):
    def _parse_entry(entry):
        items = entry.split(":")
        name = items[0] if len(items) == 2 else ""
        value = float(items[-1])
        return name, value

    return [_parse_entry(t) for t in thresholds]


class OptimizerConfig(ModelNavigatorBaseConfig):
    def _fill_config(self):
        # common config fields
        self._add_config(
            ConfigField(
                "model_name",
                flags=["--model-name"],
                field_type=ConfigPrimitive(str, required=True),
                description="Name of model.",
            )
        )

        self._add_config(
            ConfigField(
                "model_path",
                flags=["--model-path"],
                default_value="",
                field_type=ConfigPrimitive(str, required=True),
                description="Path to file with model.",
            )
        )
        self._add_config(
            ConfigField(
                "config_file",
                field_type=ConfigPrimitive(str),
                flags=["-f", "--config-file"],
                description="Path to Model Navigator Config File.",
            )
        )
        self._add_config(
            ConfigField(
                "workspace_path",
                flags=["--workspace-path"],
                default_value="workspace",
                field_type=ConfigPrimitive(str),
                description="Path to output directory.",
            )
        )
        self._add_config(
            ConfigField(
                "verbose",
                field_type=ConfigPrimitive(bool),
                parser_args={"action": "store_true"},
                default_value=False,
                flags=["--verbose"],
                description="Enable verbose mode.",
            )
        )

        # optimizer specific fields
        self._add_config(
            ConfigField(
                "target_format",
                field_type=ConfigPrimitive(str, output_mapper=parse_format),
                flags=["--target-format"],
                description="Target format to generate. If not provided, all supported formats will be generated.",
            )
        )
        self._add_config(
            ConfigField(
                "max_workspace_size",
                field_type=ConfigPrimitive(int),
                flags=["--max-workspace-size"],
                description="The amount of workspace the ICudaEngine uses.",
            )
        )
        self._add_config(
            ConfigField(
                "target_precisions",
                field_type=ConfigListString(output_mapper=parse_precisions),
                flags=["--target-precisions"],
                default_value=["fp16", "tf32"],
                description="Configure TensorRT builder for precision layer selection.",
                parser_args={"nargs": "+"},
            )
        )
        self._add_config(
            ConfigField(
                "onnx_opsets",
                field_type=ConfigListNumeric(int),
                flags=["--onnx-opsets"],
                default_value=[12, 13],
                description="Generate ONNX graph that uses only ops available in given opset.",
                parser_args={"nargs": "+"},
            )
        )

        self._add_config(
            ConfigField(
                "min_shapes",
                field_type=ConfigListString(output_mapper=parse_tensor_spec),
                flags=["--min-shapes"],
                description="The minimum shapes the TensorRT optimization profile(s) will support. "
                "Format: --min-shapes <input0>:D0,D1,..,DN .. <inputN>:D0,D1,..,DN",
                parser_args={"nargs": "*"},
            )
        )
        self._add_config(
            ConfigField(
                "opt_shapes",
                field_type=ConfigListString(output_mapper=parse_tensor_spec),
                flags=["--opt-shapes"],
                description="The optimal shapes the TensorRT optimization profile(s) will support."
                "Format: --opt-shapes <input0>:D0,D1,..,DN .. <inputN>:D0,D1,..,DN",
                parser_args={"nargs": "*"},
            )
        )
        self._add_config(
            ConfigField(
                "max_shapes",
                field_type=ConfigListString(output_mapper=parse_tensor_spec),
                flags=["--max-shapes"],
                description="The maximum shapes the TensorRT optimization profile(s) will support. "
                "Also defines shapes of input data used during performance analysis. "
                "Format: --max-shapes <input0>:D0,D1,..,DN .. <inputN>:D0,D1,..,DN",
                parser_args={"nargs": "*"},
            )
        )
        self._add_config(
            ConfigField(
                "value_ranges",
                field_type=ConfigListString(output_mapper=parse_value_range),
                flags=["--value-ranges"],
                description="Range of values used during performance analysis defined per input. "
                "Format: --value-ranges input_name0:min_value,max_value .. input_nameN:min_value,max_value",
                parser_args={"nargs": "*"},
            )
        )
        self._add_config(
            ConfigField(
                "inputs",
                field_type=ConfigListString(output_mapper=parse_tensor_spec),
                flags=["--inputs"],
                description="",
                parser_args={"nargs": "*"},
            )
        )
        self._add_config(
            ConfigField(
                "outputs",
                field_type=ConfigListString(output_mapper=parse_tensor_spec),
                flags=["--outputs"],
                description="",
                parser_args={"nargs": "*"},
            )
        )
        self._add_config(
            ConfigField(
                "rtol",
                field_type=ConfigListString(output_mapper=parse_error_thresholds),
                flags=["--rtol"],
                description="Relative tolerance parameter for output comparison. "
                "To specify per-output tolerances, use the format: --rtol [<out_name>:]<rtol>. "
                "Example: --rtol 1e-5 out0:1e-4 out1:1e-3",
                parser_args={"nargs": "*"},
            )
        )
        self._add_config(
            ConfigField(
                "atol",
                field_type=ConfigListString(output_mapper=parse_error_thresholds),
                flags=["--atol"],
                description="Absolute tolerance parameter for output comparison. "
                "To specify per-output tolerances, use the format: --atol [<out_name>:]<atol>. "
                "Example: --atol 1e-5 out0:1e-4 out1:1e-3",
                parser_args={"nargs": "*"},
            )
        )
