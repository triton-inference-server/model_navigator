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

from typing import Any, Optional

import numpy as np

from model_navigator.api.config import OptimizationProfile, SizedDataLoader, TensorRTProfile
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.core.dataloader import IndiciesFilteredDataloader, extract_sample, load_samples, samples_to_npz
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.frameworks import Framework
from model_navigator.runners.utils import get_format_default_runners
from model_navigator.utils.format_helpers import FRAMEWORK2BASE_FORMAT


class FetchInputModelData(Command, is_required=True):
    """Command for fetching input samples from the dataloader."""

    def _run(
        self,
        workspace: Workspace,
        framework: Framework,
        dataloader: SizedDataLoader,
        sample_count: int,
        input_metadata: TensorMetadata,
        batch_dim: Optional[int],
        seed: int,
        dataloader_trt_profile: TensorRTProfile,
        optimization_profile: OptimizationProfile,
        raise_on_error: Optional[bool] = False,
    ) -> CommandOutput:
        """Run the command.

        There are three types of samples that are fetched form the dataloader:
            1) profiling sample - one sample for profiling,
            2) conversion samples - samples spanning all dimensions sizes from min to max,
            3) correctness samples - `sample_count` samples for verifying correctness.

        Args:
            workspace: Workspace of current execution.
            framework: Model framework.
            dataloader: Dataloader for the model.
            sample_count: Number of correctness samples to fetch.
            input_metadata: Input metadata.
            batch_dim: Batch dimension.
            seed: Random seed.
            dataloader_trt_profile: Model TensorRT Profile.
            optimization_profile: Performance configuration with dataloader override
            raise_on_error: If True raise an error when one of the samples is invalid. Defaults to False.

        Returns:
            CommandOutput: Fetched samples.
        """
        num_samples = len(dataloader)
        if sample_count > num_samples:
            LOGGER.warning(
                f"Requested sample_count ({sample_count}) is larger than "
                f"the number of available samples ({num_samples}). Using {num_samples} samples."
            )
            sample_count = num_samples

        LOGGER.info("Collecting input samples for model.")
        np.random.seed(seed)
        correctness_samples_ind = set(np.random.choice(num_samples, size=sample_count, replace=False))
        profiling_sample_ind, conversion_samples_ind = self._collect_samples(
            dataloader,
            input_metadata,
            dataloader_trt_profile,
            framework,
            correctness_samples_ind,
        )

        sample_data_path = workspace.path / "model_input"
        sample_data_path.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Saving samples into the workspace.")
        for samples_ind, dirname in [
            ([profiling_sample_ind], "profiling"),
            (correctness_samples_ind, "correctness"),
            (conversion_samples_ind, "conversion"),
        ]:
            sample_path = sample_data_path / dirname
            if dirname == "profiling" and optimization_profile.dataloader is not None:
                LOGGER.info("Using performance dataloader for profiling sample. Collecting first item only.")
                samples = IndiciesFilteredDataloader(optimization_profile.dataloader, [0])
            else:
                samples = IndiciesFilteredDataloader(dataloader, samples_ind)
            samples_to_npz(
                samples,
                sample_path,
                batch_dim,
                metadata=input_metadata,
                framework=framework,
                raise_on_error=raise_on_error,
            )

        return CommandOutput(
            status=CommandStatus.OK,
        )

    @staticmethod
    def _collect_samples(dataloader, input_metadata, trt_profile, framework, correctness_samples_ind):
        profiling_sample_ind = None
        correctness_samples_ind = []
        conversion_samples_ind = []
        conversion_min_max_sampled = {
            name: {ax: {"min": False, "max": False} for ax in range(len(input_metadata[name].shape))}
            for name in input_metadata
        }
        for i, sample in enumerate(dataloader):
            if i >= len(dataloader):
                break
            sample = extract_sample(sample, input_metadata, framework)

            do_sample_conversion = False
            do_sample_profiling = False
            for name in input_metadata:
                if name not in trt_profile:
                    if not conversion_min_max_sampled[name]:
                        do_sample_conversion = True
                        conversion_min_max_sampled[name] = True
                else:
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
                conversion_samples_ind.append(i)
            if do_sample_profiling:
                profiling_sample_ind = i

        if not conversion_samples_ind:
            conversion_samples_ind = correctness_samples_ind[:1]
        if profiling_sample_ind is None:
            profiling_sample_ind = correctness_samples_ind[0]

        return profiling_sample_ind, conversion_samples_ind


class FetchOutputModelData(Command, is_required=True):
    """Command for saving model outputs."""

    def _run(
        self,
        framework: Framework,
        workspace: Workspace,
        model: Any,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        batch_dim: Optional[int],
        raise_on_error: Optional[bool] = True,
    ) -> CommandOutput:
        """Run the command and save model outputs.

        Args:
            framework: Model framework.
            workspace: Model Navigator workspace path.
            model: Model instance.
            input_metadata: Input metadata.
            output_metadata: Output metadata.
            batch_dim: Batch dimension.
            raise_on_error: If True raise an error when one of the samples is invalid.
                Defaults to True.

        Returns:
            CommandOutput
        """
        output_data_path = workspace.path / "model_output"
        output_data_path.mkdir(parents=True, exist_ok=True)

        runner = get_format_default_runners(FRAMEWORK2BASE_FORMAT[framework])[0](
            model=model,
            input_metadata=input_metadata,
            output_metadata=output_metadata,
        )  # pytype: disable=not-instantiable

        for input_sample, sample_name in [
            ("profiling_sample", "profiling"),
            ("correctness_samples", "correctness"),
            ("conversion_samples", "conversion"),
        ]:
            samples = load_samples(samples_name=input_sample, workspace=workspace.path, batch_dim=batch_dim)
            with runner:
                outputs = (runner.infer(sample) for sample in samples)

                sample_path = output_data_path / sample_name
                samples_to_npz(outputs, sample_path, batch_dim, raise_on_error=raise_on_error, num_samples=len(samples))

        return CommandOutput(
            status=CommandStatus.OK,
        )
