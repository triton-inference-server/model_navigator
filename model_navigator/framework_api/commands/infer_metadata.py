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

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union

from polygraphy.backend.onnxrt import SessionFromOnnx

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.correctness import Tolerance
from model_navigator.framework_api.common import Sample, SizedDataLoader, TensorMetadata
from model_navigator.framework_api.exceptions import ExecutionContext, UserError
from model_navigator.framework_api.runners.onnx import OnnxrtRunner
from model_navigator.framework_api.utils import (
    Framework,
    RuntimeProvider,
    sample_to_tuple,
    to_numpy,
    validate_sample_input,
)

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


class InferInputMetadata(Command):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(name="Infer input metadata.", command_type=CommandType.INFER_MODEL_INPUT, requires=requires)

    @staticmethod
    def get_output_name():
        return "input_metadata"

    def _update_package_descriptor(self, package_descriptor: "PackageDescriptor", **kwargs) -> None:
        package_descriptor.navigator_status.input_metadata = self.output

    def __call__(
        self,
        model: Union[object, Path],
        framework: Framework,
        dataloader: SizedDataLoader,
        onnx_runtimes: Tuple[RuntimeProvider, ...],
        _input_names: Optional[Tuple[str, ...]] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        batch_dim: Optional[int] = None,
        **kwargs,
    ) -> Tolerance:

        sample = next(iter(dataloader))
        validate_sample_input(sample, framework)
        input_tuple = sample_to_tuple(sample)
        input_names = _input_names
        if input_names is None:
            if framework == Framework.ONNX:
                # pytype: disable=attribute-error
                onnx_runner = OnnxrtRunner(SessionFromOnnx(model.as_posix(), providers=onnx_runtimes))
                # pytype: enable=attribute-error
                with onnx_runner:
                    return TensorMetadata.from_polygraphy_tensor_metadata(onnx_runner.get_input_metadata())
            elif isinstance(sample, Mapping):
                input_names = tuple(sample.keys())
            else:
                input_names = tuple(f"input__{i}" for i in range(len(input_tuple)))

        input_metadata = TensorMetadata()
        for input_name, input_tensor in zip(input_names, input_tuple):
            np_tensor = to_numpy(input_tensor, framework)
            shape = list(np_tensor.shape)
            if batch_dim is not None:
                shape[batch_dim] = -1
            if dynamic_axes is not None:
                for i in dynamic_axes.get(input_name, {}):
                    shape[i] = -1  # if isinstance(dynamic_axes[input_name], List) else dynamic_axes[input_name][i]
            input_metadata.add(input_name, tuple(shape), np_tensor.dtype)

        return input_metadata


class InferOutputMetadata(Command):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(
            name="Infer output metadata.",
            command_type=CommandType.INFER_MODEL_OUTPUT,
            requires=requires,
        )

    @staticmethod
    def get_output_name():
        return "output_metadata"

    def _update_package_descriptor(self, package_descriptor: "PackageDescriptor", **kwargs) -> None:
        package_descriptor.navigator_status.output_metadata = self.output

    def __call__(
        self,
        framework: Framework,
        profiling_sample: Sample,
        model: Union[object, Path],
        input_metadata: TensorMetadata,
        onnx_runtimes: Tuple[RuntimeProvider, ...],
        target_device: Optional[str] = None,
        _output_names: Optional[Tuple[str, ...]] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        forward_kw_names: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ) -> TensorMetadata:

        if framework == Framework.PYT:
            from model_navigator.framework_api.runners.pyt import PytRunner

            runner = PytRunner(
                model, input_metadata, _output_names, target_device=target_device, forward_kw_names=forward_kw_names
            )
        elif framework == Framework.TF2:
            from model_navigator.framework_api.runners.tf import TFRunner

            runner = TFRunner(model, input_metadata, _output_names)
        elif framework == Framework.ONNX:
            # pytype: disable=attribute-error
            runner = OnnxrtRunner(SessionFromOnnx(model.as_posix(), providers=onnx_runtimes))
            # pytype: enable=attribute-error
        else:
            raise UserError(f"Unknown framework: {framework.value}")
        with runner, ExecutionContext():
            output = runner.infer(profiling_sample)

        output_metadata = TensorMetadata()
        for output_name, output_tensor in output.items():
            shape = list(output_tensor.shape)
            if dynamic_axes is not None:
                for i in dynamic_axes.get(output_name, {}):
                    shape[i] = -1  # if isinstance(dynamic_axes[output_name], List) else dynamic_axes[output_name][i]
            output_metadata.add(output_name, tuple(shape), output_tensor.dtype)

        return output_metadata
