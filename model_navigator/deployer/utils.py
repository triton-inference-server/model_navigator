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
from .. import Accelerator, Format, Precision
from ..log import section_header
from ..model import Model
from ..resources import (
    ACCELERATOR2RESOURCE,
    FORMAT2RESOURCE,
    FORMAT_RESOURCE,
    TRITON_RESOURCES,
    Resource,
)


def prepare_log_header(model: Model) -> str:
    lines = list()
    lines.append(f"Deploying model in format {FORMAT2RESOURCE[model.format]} on Triton Inference Server.\n")
    lines.append("In case of any issue please review helpful link section to address problems correctly.")

    lines.append(section_header("Helpful links"))

    fmt_resource = FORMAT_RESOURCE.get(model.format)
    server = TRITON_RESOURCES[Resource.TRITON_SERVER]
    backend = TRITON_RESOURCES.get(model.format)

    resources = {fmt_resource, server, backend}
    # pytype: disable=attribute-error
    for resource in filter(lambda item: item is not None, resources):
        lines.append(f"{resource.name}: {resource.link}")
    # pytype: enable=attribute-error
    header = "\n".join(lines)
    header += section_header("Model Configuration")
    header += _get_model_config_info(model)

    return header


def _get_model_config_info(model: Model) -> str:
    data = {
        "Variant name": model.name,
        "Model format": _get_format_info(model.format),
        "Backend Accelerator": _get_accelerator_info(model.accelerator),
        "Maximal batch size for model": model.max_batch_size,
        "Precision (only for TensorRT accelerator)": _get_precision_info(model.precision),
        "Capture CUDA graph optimization": _get_cuda_graph_opt_info(model.capture_cuda_graph),
        "Number of model instances": model.gpu_engine_count,
    }

    config = list()
    for key, value in data.items():
        config.append(f"{key}: {value}")

    content = "\n".join(config)

    return content


def _get_accelerator_info(accelerator: Accelerator) -> str:
    if accelerator == Accelerator.NONE:
        return "-"

    return ACCELERATOR2RESOURCE[accelerator]


def _get_cuda_graph_opt_info(flag: int) -> str:
    return "Yes" if flag else "No"


def _get_format_info(format: Format) -> str:
    return FORMAT2RESOURCE[format]


def _get_precision_info(precision: Precision) -> str:
    if precision == Precision.ANY:
        return "-"

    return precision.value
