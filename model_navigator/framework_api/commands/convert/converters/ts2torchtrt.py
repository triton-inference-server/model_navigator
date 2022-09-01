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
from typing import Optional

import fire
import numpy as np
import torch  # pytype: disable=import-error

from model_navigator.converter.config import TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.framework_api.utils import numpy_to_torch_dtype


def _get_precision(precision, precision_mode):
    precision = TensorRTPrecision(precision)
    precision_mode = TensorRTPrecisionMode(precision_mode)
    if precision_mode == TensorRTPrecisionMode.HIERARCHY:
        enabled_precisions = {
            TensorRTPrecision.FP32: [torch.float],
            TensorRTPrecision.FP16: [torch.float, torch.half],
            TensorRTPrecision.INT8: [torch.float, torch.half, torch.int8],
        }[precision]
    elif precision_mode == TensorRTPrecisionMode.SINGLE:
        enabled_precisions = {
            TensorRTPrecision.FP32: [torch.float],
            TensorRTPrecision.FP16: [torch.half],
            TensorRTPrecision.INT8: [torch.int8],
        }[precision]
    else:
        raise ValueError(
            f"Unsupported precision mode {precision_mode}. Only {TensorRTPrecisionMode.HIERARCHY} and "
            f"{TensorRTPrecisionMode.SINGLE} are allowed"
        )
    return {
        "enabled_precisions": enabled_precisions,
    }


def convert(
    exported_model_path,
    converted_model_path,
    shapes,
    input_dtypes,
    max_workspace_size,
    precision,
    precision_mode,
    target_device,
    workdir: Optional[str] = None,
):
    import torch_tensorrt  # pytype: disable=import-error

    if not workdir:
        workdir = pathlib.Path.cwd()
    workdir = pathlib.Path(workdir)

    print(type(shapes), shapes)
    print(type(input_dtypes), input_dtypes)

    input_dtypes = [numpy_to_torch_dtype(np.dtype(input_dtype)) for input_dtype in input_dtypes]
    model_input_shapes = []
    for input_shapes, input_dtype in zip(shapes.values(), input_dtypes):
        model_input_shapes.append(
            torch_tensorrt.Input(
                min_shape=input_shapes["min"],
                opt_shape=input_shapes["opt"],
                max_shape=input_shapes["max"],
                dtype=input_dtype,
            )
        )

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = workdir / exported_model_path

    model = torch.jit.load(exported_model_path.as_posix(), map_location=target_device)

    tr_model_compiled = torch_tensorrt.compile(
        module=model,
        inputs=model_input_shapes,
        workspace_size=max_workspace_size,
        truncate_long_and_double=True,
        **_get_precision(precision, precision_mode),
    )

    converted_model_path = pathlib.Path(converted_model_path)
    if not converted_model_path.is_absolute():
        converted_model_path = workdir / converted_model_path

    tr_model_compiled.save(converted_model_path.as_posix())


if __name__ == "__main__":
    fire.Fire(convert)
