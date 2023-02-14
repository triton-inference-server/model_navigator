# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Commands for fetching and dumping model IO."""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy
import numpy as np

from model_navigator.api.config import Sample, SizedDataLoader, TensorRTProfile
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.logger import LOGGER
from model_navigator.utils.dataloader import extract_bs1, extract_sample
from model_navigator.utils.format_helpers import FRAMEWORK2BASE_FORMAT
from model_navigator.utils.framework import Framework
from model_navigator.utils.runners import get_format_default_runners
from model_navigator.utils.tensor import TensorMetadata


def _validate_tensor(tensor: np.ndarray, *, raise_on_error: bool = True):
    if any(np.isnan(tensor.flatten())):
        message = "Tensor data contains `NaN` value. Please verify the dataloader and model."
        if raise_on_error:
            raise ModelNavigatorUserInputError(message)
        else:
            LOGGER.warning(message)

    if any(np.isinf(tensor.flatten())):
        message = "Tensor data contains `inf` value. Please verify the dataloader and model."
        if raise_on_error:
            raise ModelNavigatorUserInputError(message)
        else:
            LOGGER.warning(message)


def samples_to_npz(samples: List[Sample], path: Path, batch_dim: Optional[int], *, raise_on_error: bool = True) -> None:
    """Save samples to .npz files. Each sample is saved to `path/{sample index}.npz` file.

    Args:
        samples (List[Sample]): Samples to save.
        path (Path): Output directory.
        batch_dim (Optional[int]): Batch dimension
        raise_on_error (bool, optional): If True raise an error when sample is invalid. Defaults to True.
    """
    path.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        squeezed_sample = {}
        for name, tensor in sample.items():
            if batch_dim is not None:
                tensor = tensor.squeeze(batch_dim)

            _validate_tensor(tensor, raise_on_error=raise_on_error)

            squeezed_sample[name] = tensor

        np.savez((path / f"{i}.npz").as_posix(), **squeezed_sample)


class FetchInputModelData(Command, is_required=True):
    """Command for fetching input samples from the dataloader."""

    def _run(
        self,
        framework: Framework,
        dataloader: SizedDataLoader,
        sample_count: int,
        input_metadata: TensorMetadata,
        batch_dim: Optional[int],
        seed: int,
        dataloader_trt_profile: TensorRTProfile,
    ) -> CommandOutput:
        """Run the command.

        There are three types of samples that are fetched form the dataloader:
            1) profiling sample - one sample for profiling,
            2) conversion samples - samples spanning all dimensions sizes from min to max,
            3) correctness samples - `sample_count` samples for veryfing correctness.

        Args:
            framework (Framework): Model framework.
            dataloader (SizedDataLoader): Dataloader for the model.
            sample_count (int): Number of correctness samples to fetch.
            input_metadata (TensorMetadata): Input metadata.
            batch_dim (Optional[int]): Batch dimension.
            seed (int): Random seed.
            trt_profile (Profile): Model TensorRT Profile.

        Returns:
            CommandOutput: Fetched samples.
        """
        num_samples = len(dataloader)
        if sample_count > num_samples:
            LOGGER.warning(
                f"Requested sample_count ({sample_count}) is larger than"
                f"the number of available samples ({num_samples}). Using {num_samples} samples."
            )
            sample_count = num_samples

        numpy.random.seed(seed)
        correctness_samples_ind = set(numpy.random.choice(num_samples, size=sample_count, replace=False))
        profiling_sample, correctness_samples, conversion_samples = self._collect_samples(
            dataloader, input_metadata, dataloader_trt_profile, framework, batch_dim, correctness_samples_ind
        )
        return CommandOutput(
            status=CommandStatus.OK,
            output={
                "profiling_sample": profiling_sample,
                "correctness_samples": correctness_samples,
                "conversion_samples": conversion_samples,
            },
        )

    @staticmethod
    def _collect_samples(dataloader, input_metadata, trt_profile, framework, batch_dim, correctness_samples_ind):
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


class DumpInputModelData(Command, is_required=True):
    """Comand for saving input samples."""

    def _run(
        self,
        workspace: Path,
        profiling_sample: Sample,
        correctness_samples: List[Sample],
        conversion_samples: List[Sample],
        batch_dim: Optional[int],
        raise_on_error: Optional[bool] = False,
    ) -> CommandOutput:
        """Run the command and save input samples.

        Args:
            workspace (Path): Model Navigator workspace path.
            profiling_sample (Sample): Profiling sample.
            correctness_samples (List[Sample]): Correctness samples.
            conversion_samples (List[Sample]): Conversion samples.
            batch_dim (Optional[int]): Batch dimension.
            raise_on_error (Optional[bool], optional): If True raise an error when one of the samples is invalid.
                Defaults to False.

        Returns:
            CommandOutput
        """
        sample_data_path = workspace / "model_input"
        sample_data_path.mkdir(parents=True, exist_ok=True)

        for samples, dirname in [
            ([profiling_sample], "profiling"),
            (correctness_samples, "correctness"),
            (conversion_samples, "conversion"),
        ]:
            samples_to_npz(samples, sample_data_path / dirname, batch_dim, raise_on_error=raise_on_error)

        return CommandOutput(status=CommandStatus.OK)


class DumpOutputModelData(Command, is_required=True):
    """Comand for saving model outputs."""

    def _run(
        self,
        framework: Framework,
        workspace: Path,
        model: Any,
        profiling_sample: Sample,
        correctness_samples: List[Sample],
        conversion_samples: List[Sample],
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        batch_dim: Optional[int],
        raise_on_error: Optional[bool] = True,
        forward_kw_names: Optional[Tuple[str, ...]] = None,
    ) -> CommandOutput:
        """Run the command and save model outputs.

        Args:
            framework (Framework): Model framework.
            workspace (Path): Model Navigator workspace path.
            model (Any): Model instance.
            profiling_sample (Sample): Profiling sample.
            correctness_samples (List[Sample]): Correctness samples.
            conversion_samples (List[Sample]): Conversion samples.
            input_metadata (TensorMetadata): Input metadata.
            output_metadata (TensorMetadata): Output metadata.
            batch_dim (Optional[int]): Batch dimension.
            raise_on_error (Optional[bool], optional): If True raise an error when one of the samples is invalid.
                Defaults to True.
            forward_kw_names (Optional[Tuple[str, ...]], optional): List of source model input signature naems.
                Defaults to None.

        Returns:
            CommandOutput
        """
        output_data_path = workspace / "model_output"
        output_data_path.mkdir(parents=True, exist_ok=True)

        runner = get_format_default_runners(FRAMEWORK2BASE_FORMAT[framework])[0](
            model=model,
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            input_metadata_mapping=forward_kw_names,
        )  # pytype: disable=not-instantiable

        ret = []
        for samples, dirname in [
            ([profiling_sample], "profiling"),
            (correctness_samples, "correctness"),
            (conversion_samples, "conversion"),
        ]:
            with runner:
                outputs = [runner.infer(sample) for sample in samples]

            samples_to_npz(outputs, output_data_path / dirname, batch_dim, raise_on_error=raise_on_error)
            ret.append(outputs)

        outputs_names = (
            "profiling_sample_output",
            "correctness_samples_output",
            "conversion_samples_output",
        )
        return CommandOutput(status=CommandStatus.OK, output=dict(zip(outputs_names, ret)))
