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
from typing import Callable, List, Optional, Tuple

import numpy

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import Sample, TensorMetadata
from model_navigator.framework_api.exceptions import TensorTypeError
from model_navigator.framework_api.utils import Framework, get_package_path, sample_to_tuple, to_numpy


# TODO: Add support for: Numpy Arrays, Dict??, tf.data??, keras.utils.Sequence??
def is_tensor(sample, framework: Framework):
    if framework == Framework.PYT:
        import torch  # pytype: disable=import-error

        tensor_check = torch.is_tensor
        expected_type = type(torch.Tensor)
    else:
        import tensorflow  # pytype: disable=import-error

        tensor_check = tensorflow.is_tensor
        expected_type = type(tensorflow.Tensor)

    if isinstance(sample, (list, tuple)):
        for tensor in sample:
            if not tensor_check(tensor):
                raise TensorTypeError(f"Expected type: {expected_type}, found: {type(tensor)}")
    elif isinstance(sample, dict):
        for tensor in sample.values():
            if not tensor_check(tensor):
                raise TensorTypeError(f"Expected type: {expected_type}, found: {type(tensor)}")
    else:
        if not tensor_check(sample):
            raise TensorTypeError(f"Expected type: {expected_type}, found: {type(sample)}")


class FetchInputModelData(Command):
    def __init__(self):
        super().__init__(
            name="Fetch input model data",
            command_type=CommandType.DUMP_MODEL_INPUT,
        )

    @staticmethod
    def get_output_name():
        return "samples"

    def __call__(
        self,
        framework: Framework,
        dataloader: Callable,
        sample_count: int,
        input_metadata: TensorMetadata,
        **kwargs,
    ) -> List[Sample]:
        generator = dataloader()
        samples = []
        for i, sample in enumerate(generator):
            if i >= sample_count:
                break
            is_tensor(sample, framework)
            sample = sample_to_tuple(sample)
            sample = {n: to_numpy(t, framework) for n, t in zip(input_metadata, sample)}
            samples.append(sample)
        return samples


class DumpInputModelData(Command):
    def __init__(self):
        super().__init__(
            name="Dump input model data",
            command_type=CommandType.DUMP_MODEL_INPUT,
        )

    @staticmethod
    def get_output_relative_path() -> Path:
        return Path("model_input")

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        samples: List,
        **kwargs,
    ):
        sample_data_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        sample_data_path.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(samples):
            sample_file_path = sample_data_path / f"sample_{i}.npz"
            numpy.savez(sample_file_path, **sample)

        return samples


class DumpOutputModelData(Command):
    def __init__(self):
        super().__init__(
            name="Dump output model data",
            command_type=CommandType.DUMP_MODEL_OUTPUT,
        )

    @staticmethod
    def get_output_relative_path() -> Path:
        return Path("model_output")

    def __call__(
        self,
        framework: Framework,
        workdir: Path,
        model,
        model_name: str,
        samples: List,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        target_device: Optional[str] = None,
        forward_kw_names: Optional[Tuple[str]] = None,
        **kwargs,
    ):
        output_names = list(output_metadata.keys())
        outputs = []
        output_data_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        output_data_path.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            if framework == Framework.PYT:
                from model_navigator.framework_api.runners.pyt import PytRunner

                runner = PytRunner(
                    model, input_metadata, output_names, target_device=target_device, forward_kw_names=forward_kw_names
                )
            else:
                from model_navigator.framework_api.runners.tf import TFRunner

                runner = TFRunner(model, input_metadata, output_names)

            with runner:
                output = runner.infer(sample)
                outputs.append(output)

        for i, output in enumerate(outputs):
            numpy.savez(output_data_path / f"sample_{i}.npz", **output)

        return self.get_output_relative_path()
