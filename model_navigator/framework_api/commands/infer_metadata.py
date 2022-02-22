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

from typing import Callable, Dict, List, Optional, Tuple, Union

from model_navigator.framework_api.commands.core import Command, CommandType, Tolerance
from model_navigator.framework_api.common import Sample, TensorMetadata
from model_navigator.framework_api.utils import Framework, sample_to_tuple, to_numpy


class InferInputMetadata(Command):
    def __init__(self):
        super().__init__(
            name="Infer input metadata.",
            command_type=CommandType.CUSTOM,
        )

    @staticmethod
    def get_output_name():
        return "input_metadata"

    def __call__(
        self,
        framework: Framework,
        dataloader: Callable,
        _input_names: Optional[Tuple[str]] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        **kwargs,
    ) -> Tolerance:

        sample = next(iter(dataloader()))
        input_tuple = sample_to_tuple(sample)
        input_names = _input_names
        if input_names is None:
            if isinstance(sample, dict):
                input_names = tuple(sample.keys())
            else:
                input_names = tuple(f"input__{i}" for i in range(len(input_tuple)))

        input_metadata = TensorMetadata()
        for input_name, input_tensor in zip(input_names, input_tuple):
            np_tensor = to_numpy(input_tensor, framework)
            shape = list(np_tensor.shape)
            if dynamic_axes is not None:
                for i in dynamic_axes.get(input_name, {}):
                    shape[i] = -1  # if isinstance(dynamic_axes[input_name], List) else dynamic_axes[input_name][i]
            input_metadata.add(input_name, tuple(shape), np_tensor.dtype)

        return input_metadata


class InferOutputMetadata(Command):
    def __init__(self):
        super().__init__(
            name="Infer output metadata.",
            command_type=CommandType.CUSTOM,
        )

    @staticmethod
    def get_output_name():
        return "output_metadata"

    def __call__(
        self,
        framework: Framework,
        samples: List[Sample],
        model: object,
        input_metadata: TensorMetadata,
        target_device: Optional[str] = None,
        _output_names: Optional[Tuple[str]] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        forward_kw_names: Optional[Tuple[str]] = None,
        **kwargs,
    ) -> Tolerance:

        if framework == Framework.PYT:
            from model_navigator.framework_api.runners.pyt import PytRunner

            runner = PytRunner(
                model, input_metadata, _output_names, target_device=target_device, forward_kw_names=forward_kw_names
            )
        else:
            from model_navigator.framework_api.runners.tf import TFRunner

            runner = TFRunner(model, input_metadata, _output_names)

        with runner:
            output = runner.infer(samples[0])

        output_metadata = TensorMetadata()
        for output_name, output_tensor in output.items():
            shape = list(output_tensor.shape)
            if dynamic_axes is not None:
                for i in dynamic_axes.get(output_name, {}):
                    shape[i] = -1  # if isinstance(dynamic_axes[output_name], List) else dynamic_axes[output_name][i]
            output_metadata.add(output_name, tuple(shape), output_tensor.dtype)

        return output_metadata
