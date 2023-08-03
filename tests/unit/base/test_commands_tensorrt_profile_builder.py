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

# TODO: Enable tests when multi-profile support is added.

# import numpy as np

# from model_navigator.api.config import TensorRTProfile
# from model_navigator.commands.base import CommandStatus
# from model_navigator.commands.performance.profiler import ProfilingResults
# from model_navigator.commands.tensorrt_profile_builder import ProfileType, TensorRTProfileBuilder


# def test_tensorrt_profile_builder_returns_valid_profiles():
#     builder = TensorRTProfileBuilder()

#     dataloader_trt_profile = TensorRTProfile().add(
#         name="input_0", min=(1, 224, 224, 3), opt=(8, 224, 224, 3), max=(16, 224, 224, 3)
#     )

#     valid_profiles = {
#         ProfileType.MAX_THROUGHPUT: TensorRTProfile().add(
#             "input_0", min=(1, 224, 224, 3), opt=(16, 224, 224, 3), max=(16, 224, 224, 3)
#         ),
#         ProfileType.MAX_THROUGHPUT_STATIC: TensorRTProfile().add(
#             "input_0", min=(16, 224, 224, 3), opt=(16, 224, 224, 3), max=(16, 224, 224, 3)
#         ),
#         ProfileType.MIN_LATENCY: TensorRTProfile().add(
#             "input_0", min=(1, 224, 224, 3), opt=(1, 224, 224, 3), max=(1, 224, 224, 3)
#         ),
#         ProfileType.OPT_LATENCY: TensorRTProfile().add(
#             "input_0", min=(1, 224, 224, 3), opt=(4, 224, 224, 3), max=(4, 224, 224, 3)
#         ),
#         ProfileType.FALLBACK: TensorRTProfile().add(
#             "input_0", min=(1, 224, 224, 3), opt=(16, 224, 224, 3), max=(32, 224, 224, 3)
#         ),
#     }

#     profiling_results = [
#         ProfilingResults.from_measurements([10, 10, 10], [1500, np.nan], 1, 0),
#         ProfilingResults.from_measurements([10, 10, 10], [1500, np.nan], 2, 0),
#         ProfilingResults.from_measurements([11, 11, 11], [1500, np.nan], 4, 0),
#         ProfilingResults.from_measurements([30, 30, 30], [1500, np.nan], 8, 0),
#         ProfilingResults.from_measurements([40, 40, 40], [1500, np.nan], 16, 0),
#         ProfilingResults.from_measurements([80, 80, 80], [1500, np.nan], 32, 0),
#     ]

#     command_output = builder.run(
#         dataloader_trt_profile=dataloader_trt_profile,
#         profiling_results=profiling_results,
#     )

#     trt_profiles = command_output.output["optimized_trt_profiles"]
#     assert len(trt_profiles) == 5
#     for trt_profile, valid_profile in zip(trt_profiles, valid_profiles.values()):
#         assert "input_0" in trt_profile
#         assert trt_profile == valid_profile


# def test_tensorrt_profile_builder_returns_valid_profiles_with_latency_budget():
#     builder = TensorRTProfileBuilder()

#     dataloader_trt_profile = TensorRTProfile().add(
#         name="input_0", min=(1, 224, 224, 3), opt=(8, 224, 224, 3), max=(16, 224, 224, 3)
#     )

#     valid_profiles = {
#         ProfileType.MAX_THROUGHPUT: TensorRTProfile().add(
#             "input_0", min=(1, 224, 224, 3), opt=(16, 224, 224, 3), max=(16, 224, 224, 3)
#         ),
#         ProfileType.MAX_THROUGHPUT_STATIC: TensorRTProfile().add(
#             "input_0", min=(16, 224, 224, 3), opt=(16, 224, 224, 3), max=(16, 224, 224, 3)
#         ),
#         ProfileType.MIN_LATENCY: TensorRTProfile().add(
#             "input_0", min=(1, 224, 224, 3), opt=(1, 224, 224, 3), max=(1, 224, 224, 3)
#         ),
#         ProfileType.OPT_LATENCY: TensorRTProfile().add(
#             "input_0", min=(1, 224, 224, 3), opt=(4, 224, 224, 3), max=(4, 224, 224, 3)
#         ),
#         ProfileType.FALLBACK: TensorRTProfile().add(
#             "input_0", min=(1, 224, 224, 3), opt=(16, 224, 224, 3), max=(32, 224, 224, 3)
#         ),
#         ProfileType.LATENCY_BUDGET: TensorRTProfile().add(
#             "input_0", min=(1, 224, 224, 3), opt=(2, 224, 224, 3), max=(2, 224, 224, 3)
#         ),
#     }

#     profiling_results = [
#         ProfilingResults.from_measurements([10, 10, 10], [1500, np.nan], 1, 0),
#         ProfilingResults.from_measurements([10, 10, 10], [1500, np.nan], 2, 0),
#         ProfilingResults.from_measurements([11, 11, 11], [1500, np.nan], 4, 0),
#         ProfilingResults.from_measurements([30, 30, 30], [1500, np.nan], 8, 0),
#         ProfilingResults.from_measurements([40, 40, 40], [1500, np.nan], 16, 0),
#         ProfilingResults.from_measurements([80, 80, 80], [1500, np.nan], 32, 0),
#     ]

#     command_output = builder.run(
#         dataloader_trt_profile=dataloader_trt_profile,
#         profiling_results=profiling_results,
#         latency_budget=10,
#     )

#     trt_profiles = command_output.output["optimized_trt_profiles"]

#     assert len(trt_profiles) == 6
#     for trt_profile, valid_profile in zip(trt_profiles, valid_profiles.values()):
#         assert "input_0" in trt_profile
#         assert trt_profile == valid_profile


# def test_tensorrt_profile_builder_return_command_status_fail_when_no_user_trt_profiles_and_no_profling_data_provided():
#     builder = TensorRTProfileBuilder()

#     dataloader_trt_profile = TensorRTProfile().add(
#         name="input_0", min=(1, 224, 224, 3), opt=(8, 224, 224, 3), max=(16, 224, 224, 3)
#     )

#     command_output = builder.run(
#         dataloader_trt_profile=dataloader_trt_profile,
#     )

#     assert command_output.status == CommandStatus.FAIL


# def test_tensorrt_profile_builder_return_user_trt_profiles_if_user_profiles_provided_by_user():
#     builder = TensorRTProfileBuilder()

#     dataloader_trt_profile = TensorRTProfile().add(
#         name="input_0", min=(1, 224, 224, 3), opt=(8, 224, 224, 3), max=(16, 224, 224, 3)
#     )

#     user_profiles = [
#         TensorRTProfile().add("input_0", min=(1, 224, 224, 3), opt=(16, 224, 224, 3), max=(16, 224, 224, 3)),
#         TensorRTProfile().add("input_0", min=(1, 224, 224, 3), opt=(1, 224, 224, 3), max=(1, 224, 224, 3)),
#         TensorRTProfile().add("input_0", min=(1, 224, 224, 3), opt=(4, 224, 224, 3), max=(4, 224, 224, 3)),
#         TensorRTProfile().add("input_0", min=(1, 224, 224, 3), opt=(2, 224, 224, 3), max=(2, 224, 224, 3)),
#         TensorRTProfile().add("input_0", min=(1, 224, 224, 3), opt=(16, 224, 224, 3), max=(32, 224, 224, 3)),
#     ]

#     command_output = builder.run(
#         dataloader_trt_profile=dataloader_trt_profile,
#         trt_profiles=user_profiles,
#     )

#     trt_profiles = command_output.output["optimized_trt_profiles"]

#     for trt_profile, user_profile in zip(trt_profiles, user_profiles):
#         assert "input_0" in trt_profile
#         assert trt_profile == user_profile
