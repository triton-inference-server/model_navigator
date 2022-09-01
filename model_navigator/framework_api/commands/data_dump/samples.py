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
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy
from polygraphy.backend.onnxrt import SessionFromOnnx
from polygraphy.backend.trt import Profile

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import Sample, SizedDataLoader, TensorMetadata
from model_navigator.framework_api.exceptions import UserError
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import (
    Framework,
    extract_bs1,
    extract_sample,
    get_available_onnx_providers,
)


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


class FetchInputModelData(Command):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(name="Fetch input model data", command_type=CommandType.FETCH_MODEL_INPUT, requires=requires)

    @staticmethod
    def get_output_name():
        return (
            "profiling_sample",
            "correctness_samples",
            "conversion_samples",
        )

    @staticmethod
    def collect_samples(dataloader, input_metadata, trt_profile, framework, batch_dim, correctness_samples_ind):
        profiling_sample = None
        correctness_samples = []
        conversion_samples = []
        conversion_min_max_sampled = {
            name: {ax: {"min": False, "max": False} for ax in range(len(input_metadata[name].shape))}
            for name in input_metadata
        }
        for i, sample in enumerate(dataloader):
            if i >= len(dataloader):
                break
            sample = extract_sample(sample, input_metadata, framework)

            if i in correctness_samples_ind:
                correctness_samples.append(extract_bs1(sample, batch_dim))
            do_sample_conversion = False
            do_sample_profiling = False
            for name in input_metadata:
                for (ax, shapes), tensor_dim in zip(
                    enumerate(zip(trt_profile[name].min, trt_profile[name].opt, trt_profile[name].max)),
                    sample[name].shape,
                ):
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
        trt_profile: Profile,
        **kwargs,
    ) -> Tuple[List[Sample], List[Sample], List[Sample]]:

        num_samples = len(dataloader)
        if sample_count > num_samples:
            LOGGER.warning(
                f"Requested sample_count ({sample_count}) is larger than the number of available samples ({num_samples}). Using {num_samples} samples."
            )
            sample_count = num_samples

        numpy.random.seed(seed)
        correctness_samples_ind = set(numpy.random.choice(num_samples, size=sample_count, replace=False))
        profiling_sample, correctness_samples, conversion_samples = self.collect_samples(
            dataloader, input_metadata, trt_profile, framework, batch_dim, correctness_samples_ind
        )

        return profiling_sample, correctness_samples, conversion_samples


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
        sample_data_path = workdir / self.get_output_relative_path()
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

    @staticmethod
    def get_output_name():
        return (
            "profiling_sample_output",
            "correctness_samples_output",
            "conversion_samples_output",
        )

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
        model_params: Optional[Any] = None,
        target_device: Optional[str] = None,
        forward_kw_names: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ):
        output_names = list(output_metadata.keys())
        output_data_path = workdir / self.get_output_relative_path()
        output_data_path.mkdir(parents=True, exist_ok=True)

        if framework == Framework.PYT:
            from model_navigator.framework_api.runners.pyt import PytRunner

            runner = PytRunner(
                model, input_metadata, output_names, target_device=target_device, forward_kw_names=forward_kw_names
            )
        elif framework == Framework.TF2:
            from model_navigator.framework_api.runners.tf import TFKerasRunner

            runner = TFKerasRunner(model, input_metadata, output_names, forward_kw_names=forward_kw_names)

        elif framework == Framework.ONNX:
            from model_navigator.framework_api.runners.onnx import OnnxrtRunner

            runner = OnnxrtRunner(
                SessionFromOnnx(model.as_posix(), providers=get_available_onnx_providers(exclude_trt=True))
            )
        elif framework == Framework.JAX:
            from model_navigator.framework_api.runners.jax import JAXRunner

            runner = JAXRunner(model, model_params, input_metadata, output_names, forward_kw_names=forward_kw_names)
        else:
            raise UserError(f"Unknown framework: {framework.value}")

        ret = []
        for samples, dirname in [
            ([profiling_sample], "profiling"),
            (correctness_samples, "correctness"),
            (conversion_samples, "conversion"),
        ]:
            with runner:
                outputs = [runner.infer(sample) for sample in samples]

            samples_to_npz(outputs, output_data_path / dirname, batch_dim)
            ret.append(outputs)

        return tuple(ret)
