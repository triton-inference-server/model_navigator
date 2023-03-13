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
import pathlib
import tempfile

from model_navigator.api.config import DeviceKind, ProfilerConfig
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.constants import NAVIGATOR_PACKAGE_VERSION, NAVIGATOR_VERSION
from model_navigator.pipelines.pipeline_manager import PipelineManager
from model_navigator.utils.framework import Framework
from tests.unit.base.mocks.packages import torchscript_package_with_torch_tensorrt


def test_prepare_package_create_new_package_when_no_package_provided(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        config = CommonConfig(
            framework=Framework.TORCH,
            dataloader=[{"input_name": [idx]} for idx in range(10)],
            workspace=workspace,
            model=None,
            profiler_config=ProfilerConfig(),
            runner_names=(),
            sample_count=10,
            target_formats=(),
            target_device=DeviceKind.CUDA,
        )

        spy_new = mocker.spy(PipelineManager, "_new_package")

        package = PipelineManager._prepare_package(
            package=None,
            config=config,
        )

        assert spy_new.called is True
        assert package is not None
        assert package.status is not None


def test_prepare_package_update_status_when_existing_package_provided(mocker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        config = CommonConfig(
            framework=Framework.TORCH,
            dataloader=[{"input_name": [idx]} for idx in range(10)],
            workspace=workspace,
            model=None,
            profiler_config=ProfilerConfig(),
            runner_names=(),
            sample_count=10,
            target_formats=(),
            target_device=DeviceKind.CUDA,
        )

        previous_package = torchscript_package_with_torch_tensorrt(workspace)

        spy_from_package = mocker.spy(PipelineManager, "_from_package")

        package = PipelineManager._prepare_package(
            package=previous_package,
            config=config,
        )

        assert spy_from_package.called is True

        assert previous_package.status.uuid != package.status.uuid
        assert package.status.models_status == {}
        assert package.status.config == {}
        assert package.status.format_version == NAVIGATOR_PACKAGE_VERSION
        assert package.status.model_navigator_version == NAVIGATOR_VERSION
