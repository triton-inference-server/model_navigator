# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Profiling utilities."""

from typing import Any, Optional


def is_throughput_saturated(
    profiling_result: Any,
    prev_profiling_result: Optional[Any],
    throughput_cutoff_threshold: Optional[float],
) -> bool:
    """Validate if throughput saturated between consecutive samples.

    Args:
        profiling_result: Current profiling results.
        prev_profiling_result: Previous profiling results.
        throughput_cutoff_threshold: Minimum throughput increase to continue profiling.

    Returns:
        True when throughput saturated between consecutive samples. False when verification disabled or not yet saturated.
    """
    if prev_profiling_result is None:
        return False

    if throughput_cutoff_threshold is None:
        return False

    return profiling_result.throughput < prev_profiling_result.throughput * (1 + throughput_cutoff_threshold)
