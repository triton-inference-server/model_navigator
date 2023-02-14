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
from typing import Dict, List

from model_navigator.commands.performance.performance import Performance
from model_navigator.core.status import Status


class ValidationError(Exception):
    pass


def collect_status(status: Status) -> Dict:
    test_status = {}
    for model_key, models_status in status.models_status.items():
        for runner_name, runner_status in models_status.runners_status.items():
            key = f"{model_key}.{runner_name}"
            test_status[key] = runner_status.status[Performance.__name__].name

    return test_status


def validate_status(status: Dict, expected_statuses: List) -> None:
    current_statuses = set(status.keys())

    if not all([expected_status in current_statuses for expected_status in expected_statuses]):
        raise ValidationError(
            """Expected statuses not match current statuses.\n """
            f"""Expected: {expected_statuses}\n"""
            f"""Current: {current_statuses}\n"""
        )
