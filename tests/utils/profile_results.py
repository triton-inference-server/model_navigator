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

import json

from model_analyzer.state.analyzer_state import AnalyzerState

from model_navigator.results import ResultsStore


def get_profile_results(workspace):
    results_store = ResultsStore(workspace)
    command_results = results_store.load("profile")
    # for profile there is single result object
    checkpoint_path = command_results[0].profiling_results_path
    with checkpoint_path.open("r") as checkpoint_file:
        state = AnalyzerState.from_dict(json.load(checkpoint_file))
    profiling_results = state.get("ResultManager.results")
    return profiling_results
