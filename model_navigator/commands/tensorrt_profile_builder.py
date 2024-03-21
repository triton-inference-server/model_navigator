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
"""Definition of TensorRT profile builders."""

from enum import Enum
from typing import List, Optional

from model_navigator.api.config import TensorRTProfile
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.performance.performance import ProfilingResults
from model_navigator.core.constants import (
    DEFAULT_PROFILING_LATENCY_CUTOFF_THRESHOLD,
    DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD,
)
from model_navigator.core.logger import LOGGER
from model_navigator.frameworks.tensorrt import utils as tensorrt_utils


class ProfileType(Enum):
    """Types of TensorRT profiles support by automatic profile search in Model Navigator.

    Profile type defines min, opt and max shapes for TensorRT model inputs.

    Args:
        MIN_LATENCY: Profile with minimum latency, min = 1, opt = 1, max = 1
        OPT_LATENCY: Profile with optimal latency. Shapes are calculated based on the latency cutoff threshold.
            This profile gives maximal throughput with latency below the cutoff threshold.
            Default latency cutoff threshold is equal to 0.1 (10%) and it's mean that if latency increase by at least 10% then threshold is exceeded.
            Latency cutoff threshold is used to find profile producing model with optimal latency. Shapes larger that this one cause latency to rapidly increase (more than 10%).
        LATENCY_BUDGET: Profile with latency constrained by latency budget.
            This profiles gives maximal throughput with latency below the latency budget.
        MAX_THROUGHPUT: Profile optimized for high throughput. Shapes are calculated based on the throughput cutoff threshold equal to 0.05 (5%).
            This profile is selected based on comparison of current batch size throughput and next batch size throughput and the difference must be smaller than 5% to select current batch size.
        MAX_THROUGHPUT_STATIC: Profile optimized for high throughput. Similar to MAX_THROUGHPUT except all shapes (min, opt, max) are equal.
        FALLBACK: Profile with fallback shape. Safe shape that tried to utilize all memory available on the device.
    """

    MIN_LATENCY = "min_latency"
    OPT_LATENCY = "opt_latency"
    LATENCY_BUDGET = "latency_budget"
    MAX_THROUGHPUT = "max_throughput"
    MAX_THROUGHPUT_STATIC = "max_throughput_static"
    FALLBACK = "fallback"


class TensorRTProfileBuilder(Command):
    """TensorRT profile builder."""

    def _run(
        self,
        dataloader_trt_profile: TensorRTProfile,
        batch_dim: int = 0,
        profiling_results: Optional[List[ProfilingResults]] = None,
        trt_profiles: Optional[List[TensorRTProfile]] = None,
        latency_budget: Optional[float] = None,
    ):
        if trt_profiles:
            LOGGER.info("Using TensorRT profiles provided by the user.")
        elif profiling_results:
            trt_profiles = self.get_profiles(
                dataloader_trt_profile=dataloader_trt_profile,
                batch_dim=batch_dim,
                profiling_results=profiling_results,
                latency_budget=latency_budget,
            )
        elif batch_dim is None:
            LOGGER.info("Batching disabled. Skipping generation of optimized TensorRT profiles.")
            return CommandOutput(status=CommandStatus.SKIPPED)
        else:
            LOGGER.error("No profiling results found.")
            return CommandOutput(status=CommandStatus.FAIL)

        profiles_string = "\n".join([str(p) for p in trt_profiles])
        LOGGER.info(f"TensorRT profiles:\n{profiles_string}")
        return CommandOutput(status=CommandStatus.OK, output={"optimized_trt_profiles": trt_profiles})

    def get_profiles(
        self,
        dataloader_trt_profile: TensorRTProfile,
        batch_dim: int,
        profiling_results: List[ProfilingResults],
        throughput_cutoff_threshold: float = DEFAULT_PROFILING_THROUGHPUT_CUTOFF_THRESHOLD,
        latency_cutoff_threshold: float = DEFAULT_PROFILING_LATENCY_CUTOFF_THRESHOLD,
        latency_budget: Optional[float] = None,
    ):
        """Get TensorRT profiles based on profiling results.

        Method iterate profiling results and compares average latency and batch sizes.
        It searches for batch sizes that matches the following criteria:
            - Minimal latency: (min = 1, opt = 1, max = 1) lowest possible latency.

            - Optimal latency: (min = 1, opt = BS, max = BS) where BS is the batch size where:
                average latency of the current batch size is less than 10% higher than average latency of the previous batch size.

            - Latency budget: (min = 1, opt = BS, max = BS) where BS is the batch size where:
                average latency of the current batch size is below the latency budget provided by the user.

            - Maximal throughput: (min = 1, opt = BS, max = BS) where BS is the batch size where:
                throughput of the current batch size is higher than throughput of the previous batch size by at least 5%.

            - Maximal throughput static: (min = BS, opt = BS, max = BS) where BS the same as Maximal throughput,
                but all shapes (min, opt, max) are equal.

            - Fallback: (min = 1, opt = BS, max = BS) where BS is the maximal batch size that saturates the device memory.


        Args:
            dataloader_trt_profile: TensorRT profile obtained from dataloader. Used as base profile for all profiles.
            batch_dim: Batch dimension index.
            profiling_results: List of profiling results.
            throughput_cutoff_threshold: Throughput cutoff threshold used for finding optimal shape saturating the device memory.
            latency_cutoff_threshold: Latency cutoff threshold used for finding optimal shape with optimal latency.
            latency_budget: Latency budget in ms. Used for finding shape with latency below the budget.
        """
        LOGGER.info("Generating optimized TensorRT profiles")
        LOGGER.info(f"Batch dimension index: {batch_dim}")
        LOGGER.info(f"Using profile generated from dataloader as base profile: {str(dataloader_trt_profile)}")

        # TODO: Enable when multi-profile support is added.
        fallback_throughput_profiling_result = profiling_results[-1]
        current_throughput_profiling_result = profiling_results[0]
        current_latency_profiling_result = profiling_results[0]
        current_latency_budget_profiling_result = profiling_results[0]
        for profiling_result in profiling_results[1:]:
            if profiling_result.throughput > current_throughput_profiling_result.throughput * (
                1 + throughput_cutoff_threshold
            ):
                current_throughput_profiling_result = profiling_result
            if profiling_result.avg_latency <= current_latency_profiling_result.avg_latency * (
                1 + latency_cutoff_threshold
            ):
                current_latency_profiling_result = profiling_result
            if latency_budget and profiling_result.avg_latency <= latency_budget:
                current_latency_budget_profiling_result = profiling_result

        trt_profiles_dict = {}
        profiles = {
            # ProfileType.MAX_THROUGHPUT: current_throughput_profiling_result.batch_size,
            # ProfileType.MAX_THROUGHPUT_STATIC: current_throughput_profiling_result.batch_size,
            # ProfileType.MIN_LATENCY: 1,
            # ProfileType.OPT_LATENCY: current_latency_profiling_result.batch_size,
            # TODO: Enable when multi-profile support is added.
            ProfileType.FALLBACK: fallback_throughput_profiling_result.batch_size,
        }

        if latency_budget:
            profiles[ProfileType.LATENCY_BUDGET] = current_latency_budget_profiling_result.batch_size

        for profile_name, batch_size in profiles.items():
            if profile_name == ProfileType.MAX_THROUGHPUT_STATIC:
                trt_profiles_dict[profile_name] = tensorrt_utils.get_new_profile_with_static_batch_size(
                    trt_profile=dataloader_trt_profile, batch_size=batch_size, batch_dim=batch_dim
                )
            else:
                trt_profiles_dict[profile_name] = tensorrt_utils.get_trt_profile_with_new_max_batch_size(
                    trt_profile=dataloader_trt_profile, max_batch_size=batch_size, batch_dim=batch_dim
                )

        return list(trt_profiles_dict.values())
