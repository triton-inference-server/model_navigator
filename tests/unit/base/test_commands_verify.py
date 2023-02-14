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

import model_navigator as nav
from model_navigator.commands.verification.verify import VerifyModel
from tests.unit.base.mocks.packages import trochscript_package_with_source


def test_verify_model_pre_run_returns_true_when_correctness_and_performance_are_ok():
    status = trochscript_package_with_source(MagicMock()).status
    verify_model = VerifyModel(status)

    runner_mock = MagicMock()
    runner_mock.name = MagicMock(return_value="TorchCPU")
    assert verify_model._pre_run(verify_func=MagicMock(), key="torch", runner_cls=runner_mock) is True


def test_verify_model_pre_run_returns_false_when_correctness_and_performance_are_not_ok():
    status = trochscript_package_with_source(MagicMock()).status
    status.models_status["torch"].runners_status["TorchCPU"].status["Correctness"] = nav.CommandStatus.FAIL
    verify_model = VerifyModel(status)

    runner_mock = MagicMock()
    runner_mock.name = MagicMock(return_value="TorchCPU")
    assert verify_model._pre_run(verify_func=MagicMock(), key="torch", runner_cls=runner_mock) is False
