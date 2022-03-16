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

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import Sample, SizedDataLoader, TensorMetadata
from model_navigator.framework_api.exceptions import TensorTypeError
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import Framework, extract_bs1, get_package_path, sample_to_tuple, to_numpy


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
    elif isinstance(sample, Mapping):
        for tensor in sample.values():
            if not tensor_check(tensor):
                raise TensorTypeError(f"Expected type: {expected_type}, found: {type(tensor)}")
    else:
        if not tensor_check(sample):
            raise TensorTypeError(f"Expected type: {expected_type}, found: {type(sample)}")


def extract_sample(sample, input_metadata, framework):
    is_tensor(sample, framework)
    sample = sample_to_tuple(sample)
    sample = {n: to_numpy(t, framework) for n, t in zip(input_metadata, sample)}
    return sample


def samples_to_json(samples: List[Sample], path: Path, batch_dim: Optional[int]) -> None:
    flatten_samples = []
    for sample in samples:
        sample_data = {}
        for name, tensor in sample.items():
            if batch_dim is not None:
                tensor = tensor.squeeze(batch_dim)
            sample_data[name] = {
                "content": tensor.flatten().tolist(),
                "shape": tensor.shape,
            }
        flatten_samples.append(sample_data)

    with open(path.as_posix(), "w") as f:
        json.dump({"data": flatten_samples}, f)


def samples_to_npz(samples: List[Sample], path: Path, batch_dim: Optional[int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        squeezed_sample = {}
        for name, tensor in sample.items():
            squeezed_sample[name] = tensor.squeeze(batch_dim) if batch_dim is not None else tensor
        numpy.savez((path / f"{i}.npz").as_posix(), **squeezed_sample)


def extract_trt_axes(axes_shapes, batch_dim):
    trt_dynamic_axes = {}
    for name, axes in axes_shapes.items():
        trt_dynamic_axes[name] = {}
        for ax, shapes in axes.items():
            if ax == batch_dim:  # min bs = 1
                trt_dynamic_axes[name][ax] = (1, int(numpy.median(shapes)), max(shapes))
            else:
                trt_dynamic_axes[name][ax] = (min(shapes), int(numpy.median(shapes)), max(shapes))
    return trt_dynamic_axes


def extract_dynamic_axes(trt_dynamic_axes):
    dynamic_axes = defaultdict(list)
    for name, axes in trt_dynamic_axes.items():
        for ax, shapes in axes.items():
            if shapes[0] != shapes[2]:  # min != max
                dynamic_axes[name].append(ax)
    return dynamic_axes


class FetchInputModelData(Command):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(name="Fetch input model data", command_type=CommandType.DUMP_MODEL_INPUT, requires=requires)

    @staticmethod
    def get_output_name():
        return (
            "profiling_sample",
            "correctness_samples",
            "conversion_samples",
            "dynamic_axes",
            "trt_dynamic_axes",
            "max_batch_size",
        )

    @staticmethod
    def collect_samples(
        dataloader, dynamic_axes, trt_dynamic_axes, input_metadata, framework, batch_dim, correctness_samples_ind
    ):
        profiling_sample = None
        correctness_samples = []
        conversion_samples = []
        conversion_min_max_sampled = {
            name: {ax: {"min": False, "max": False} for ax in trt_dynamic_axes[name]} for name in trt_dynamic_axes
        }
        for i, sample in enumerate(dataloader):
            sample = extract_sample(sample, input_metadata, framework)

            if i in correctness_samples_ind:
                correctness_samples.append(extract_bs1(sample, batch_dim))
            do_sample_conversion = False
            do_sample_profiling = False
            for name in input_metadata:
                if name not in dynamic_axes:
                    continue
                for (ax, shapes), tensor_dim in zip(trt_dynamic_axes[name].items(), sample[name].shape):
                    if ax == batch_dim:
                        continue
                    if tensor_dim == shapes[0] and not conversion_min_max_sampled[name][ax]["min"]:
                        do_sample_conversion = True
                        conversion_min_max_sampled[name][ax]["min"] = True
                    if tensor_dim == shapes[2] and not conversion_min_max_sampled[name][ax]["max"]:
                        do_sample_conversion = True
                        conversion_min_max_sampled[name][ax]["max"] = True
                        do_sample_profiling = True

            if do_sample_conversion:
                conversion_samples.append(extract_bs1(sample, batch_dim))
            if do_sample_profiling:
                profiling_sample = extract_bs1(sample, batch_dim)

        if not conversion_samples:
            conversion_samples = correctness_samples[:1]
        if profiling_sample is None:
            profiling_sample = conversion_samples[0]

        return profiling_sample, correctness_samples, conversion_samples

    def __call__(
        self,
        framework: Framework,
        dataloader: SizedDataLoader,
        sample_count: int,
        input_metadata: TensorMetadata,
        batch_dim: Optional[int],
        seed: int,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        trt_dynamic_axes: Optional[Dict[str, Dict[int, Tuple[int, int, int]]]] = None,
        **kwargs,
    ) -> List[Sample]:

        num_samples = len(dataloader)
        if sample_count > num_samples:
            LOGGER.warning(
                f"Requested sample_count ({sample_count}) is larger than the number of available samples ({num_samples}). Using {num_samples} samples."
            )
            sample_count = num_samples

        numpy.random.seed(seed)
        correctness_samples_ind = set(numpy.random.choice(num_samples, size=sample_count, replace=False))

        axes_shapes = {name: {ax: [] for ax in range(len(spec.shape))} for name, spec in input_metadata.items()}
        max_batch_size = -1
        for _, sample in enumerate(dataloader):
            sample = extract_sample(sample, input_metadata, framework)
            for name, tensor in sample.items():
                for i, dim in enumerate(tensor.shape):
                    axes_shapes[name][i].append(dim)
            if batch_dim is not None:
                max_batch_size = max(max_batch_size, sample_to_tuple(sample)[0].shape[batch_dim])

        if trt_dynamic_axes is None:
            trt_dynamic_axes = extract_trt_axes(axes_shapes, batch_dim)

            LOGGER.warning(
                f"No TRT (min, opt, max) values for axes provided. Using values derived from the dataloader: {trt_dynamic_axes}."
            )

        if dynamic_axes is None:
            dynamic_axes = extract_dynamic_axes(trt_dynamic_axes)

            LOGGER.warning(f"No dynamic axes provided. Using values derived from the dataloader: {dynamic_axes}")

        # profiling - one sample with max shape
        # correctness - sample_count random samples
        # conversion - samples with min / max shapes for all dims (except batch)
        # all samples with batch size 1
        profiling_sample, correctness_samples, conversion_samples = self.collect_samples(
            dataloader, dynamic_axes, trt_dynamic_axes, input_metadata, framework, batch_dim, correctness_samples_ind
        )

        return profiling_sample, correctness_samples, conversion_samples, dynamic_axes, trt_dynamic_axes, max_batch_size


class DumpInputModelData(Command):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(name="Dump input model data", command_type=CommandType.DUMP_MODEL_INPUT, requires=requires)

    @staticmethod
    def get_output_relative_path() -> Path:
        return Path("model_input")

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        profiling_sample: Sample,
        correctness_samples: List[Sample],
        conversion_samples: List[Sample],
        batch_dim: Optional[int],
        **kwargs,
    ):
        sample_data_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        sample_data_path.mkdir(parents=True, exist_ok=True)

        for samples, dirname in [
            ([profiling_sample], "profiling"),
            (correctness_samples, "correctness"),
            (conversion_samples, "conversion"),
        ]:
            samples_to_npz(samples, sample_data_path / dirname, batch_dim)


class DumpOutputModelData(Command):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(name="Dump output model data", command_type=CommandType.DUMP_MODEL_OUTPUT, requires=requires)

    @staticmethod
    def get_output_relative_path() -> Path:
        return Path("model_output")

    def __call__(
        self,
        framework: Framework,
        workdir: Path,
        model,
        model_name: str,
        profiling_sample: Sample,
        correctness_samples: List[Sample],
        conversion_samples: List[Sample],
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        batch_dim: Optional[int],
        target_device: Optional[str] = None,
        forward_kw_names: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ):
        output_names = list(output_metadata.keys())
        output_data_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        output_data_path.mkdir(parents=True, exist_ok=True)

        if framework == Framework.PYT:
            from model_navigator.framework_api.runners.pyt import PytRunner

            runner = PytRunner(
                model, input_metadata, output_names, target_device=target_device, forward_kw_names=forward_kw_names
            )
        else:
            from model_navigator.framework_api.runners.tf import TFRunner

            runner = TFRunner(model, input_metadata, output_names)

        for samples, dirname in [
            ([profiling_sample], "profiling"),
            (correctness_samples, "correctness"),
            (conversion_samples, "conversion"),
        ]:
            with runner:
                outputs = [runner.infer(sample) for sample in samples]
            samples_to_npz(outputs, output_data_path / dirname, batch_dim)

        return self.get_output_relative_path()
