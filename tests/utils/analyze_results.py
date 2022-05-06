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
import csv

from model_navigator.results import ResultsStore


def get_analyze_results(workspace):
    results_store = ResultsStore(workspace)
    command_results = results_store.load("analyze")
    # for profile there is single result object
    results_path = command_results[0].results_path
    with results_path.open("r") as file:
        reader = csv.DictReader(file)
        results = list(reader)

    return results
