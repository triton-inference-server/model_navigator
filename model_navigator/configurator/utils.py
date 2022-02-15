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
import pathlib
from typing import List, Union

from model_navigator.configurator.variant import Variant
from model_navigator.model import Format, Model
from model_navigator.triton.config import BackendAccelerator, TensorRTOptPrecision
from model_navigator.utils.resources import (
    ACCELERATOR2RESOURCE,
    FORMAT2RESOURCE,
    FORMAT_RESOURCE,
    TRITON_RESOURCES,
    Resource,
)


def log_configuration_error(
    workspace: Union[str, pathlib.Path], model: Model, variant: Variant, server_log: str, errors: List
) -> pathlib.Path:
    logger = FileLogger(name=variant.name, workspace=workspace)
    header = _prepare_log_header(model, variant=variant)
    logger.log(header)

    logger.log(_section_header("Client Error Log"))
    for error in errors:
        if error is not None:
            logger.log(error)

    logger.log(_section_header("Triton Inference Server Log"))
    logger.log(server_log)

    return logger.file_path


class FileLogger:
    def __init__(self, workspace: Union[str, pathlib.Path], name: str):
        filename = f"{name}.log"
        self.file_path = self.get_logs_dir(workspace) / filename

    def log(self, content: str):
        with open(self.file_path, "a+") as f:
            f.write(content)

    @classmethod
    def get_logs_dir(cls, workspace: Union[str, pathlib.Path]):
        log_dir = pathlib.Path(workspace) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir


def _prepare_log_header(source_model: Model, variant: Variant) -> str:
    lines = []
    lines.append(f"Deploying model in format {FORMAT2RESOURCE[source_model.format]} on Triton Inference Server.\n")
    lines.append("In case of any issue please review helpful link section to address problems correctly.")

    lines.append(_section_header("Helpful links"))

    fmt_resource = FORMAT_RESOURCE.get(source_model.format)
    server = TRITON_RESOURCES[Resource.TRITON_SERVER]
    backend = TRITON_RESOURCES.get(source_model.format)

    resources = {fmt_resource, server, backend}
    # pytype: disable=attribute-error
    for resource in filter(lambda item: item is not None, resources):
        lines.append(f"{resource.name}: {resource.link}")
    # pytype: enable=attribute-error
    header = "\n".join(lines)
    header += _section_header("Model Configuration")
    header += _get_model_config_info(
        source_model=source_model,
        variant=variant,
    )

    return header


def _section_header(section_name: str) -> str:
    return f"\n\n================== {section_name} ==================\n\n"


def _get_model_config_info(source_model: Model, variant: Variant) -> str:
    data = {
        "Variant name": variant.name,
        "Model format": _get_format_info(source_model.format),
        "Backend Accelerator": _get_accelerator_info(variant.optimization_config.backend_accelerator),
        "Precision (only for TensorRT accelerator)": _get_precision_info(
            variant.optimization_config.tensorrt_precision
        ),
        "Capture CUDA graph optimization": _get_cuda_graph_opt_info(
            variant.optimization_config.tensorrt_capture_cuda_graph
        ),
    }

    config = []
    for key, value in data.items():
        config.append(f"{key}: {value}")

    content = "\n".join(config)

    return content


def _get_accelerator_info(accelerator: BackendAccelerator) -> str:
    if not accelerator:
        return "-"

    return ACCELERATOR2RESOURCE[accelerator]


def _get_cuda_graph_opt_info(flag: int) -> str:
    return "Yes" if flag else "No"


def _get_format_info(format: Format) -> str:
    return FORMAT2RESOURCE[format]


def _get_precision_info(precision: TensorRTOptPrecision) -> str:
    if not precision:
        return "-"

    return precision.value
