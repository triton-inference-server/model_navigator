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
from unittest.mock import MagicMock

from model_navigator.commands.verification.verify import VerifyModel


def test_verify_model_pre_run_returns_true_when_verify_function_is_passed():
    verify_model = VerifyModel()

    runner_mock = MagicMock()
    runner_mock.name = MagicMock(return_value="TorchCPU")
    assert verify_model._pre_run(verify_func=MagicMock()) is True  # pytype: disable=wrong-arg-types


def test_verify_model_pre_run_returns_false_when_verify_function_is_none():
    verify_model = VerifyModel()

    runner_mock = MagicMock()
    runner_mock.name = MagicMock(return_value="TorchCPU")
    assert verify_model._pre_run(verify_func=None) is False  # pytype: disable=wrong-arg-types
