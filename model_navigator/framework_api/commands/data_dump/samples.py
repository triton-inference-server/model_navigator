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

from pathlib import Path
from typing import Callable, List

import numpy

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.exceptions import TensorTypeError
from model_navigator.framework_api.utils import Framework, get_package_path, model_forward_torch, to_numpy


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
        **kwargs,
    ):
        generator = dataloader()
        samples = []
        for i, sample in enumerate(generator):
            if i >= sample_count:
                break
            is_tensor(sample, framework)
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
        framework: Framework,
        workdir: Path,
        model_name: str,
        samples: List,
        sample_count: int,
        **kwargs,
    ):
        sample_data_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        sample_data_path.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(samples):
            sample_file_path = sample_data_path / f"sample_{i}.npz"
            if isinstance(sample, (list, tuple)):
                sample = [to_numpy(item, framework) for item in sample]
                numpy.savez(sample_file_path, *sample)
            elif isinstance(sample, dict):
                sample = {name: to_numpy(tensor, framework) for name, tensor in sample.items()}
                numpy.savez(sample_file_path, **sample)
            else:
                sample = to_numpy(sample, framework)
                numpy.savez(sample_file_path, sample)

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
        sample_count: int,
        **kwargs,
    ):
        outputs = []
        output_data_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        output_data_path.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            if framework == Framework.PYT:
                import torch  # pytype: disable=import-error

                inputs = sample
                output = model_forward_torch(model, inputs)
                if isinstance(output, (list, tuple)) and isinstance(output[0], torch.Tensor):
                    output = [to_numpy(tensor, framework) for tensor in output]
                elif isinstance(output, dict):
                    output = {name: to_numpy(tensor, framework) for name, tensor in output.items()}
                elif isinstance(output, torch.Tensor):
                    output = to_numpy(output, framework)
                outputs.append(output)
            else:
                outputs.append(to_numpy(model.predict(sample), framework))

        for i, output in enumerate(outputs):
            if isinstance(output, (list, tuple)):
                numpy.savez(output_data_path / f"sample_{i}.npz", *output)
            elif isinstance(output, dict):
                numpy.savez(output_data_path / f"sample_{i}.npz", **output)
            else:
                numpy.savez(output_data_path / f"sample_{i}.npz", output)

        return self.get_output_relative_path()
