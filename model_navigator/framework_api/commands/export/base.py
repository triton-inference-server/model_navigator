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

from typing import TYPE_CHECKING, List

from model_navigator.framework_api.commands.core import Command
from model_navigator.framework_api.status import ModelStatus, RuntimeResults
from model_navigator.framework_api.utils import RuntimeProvider, format2runtimes
from model_navigator.model import Format

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


class ExportBase(Command):
    def _update_package_descriptor(
        self, package_descriptor: "PackageDescriptor", onnx_runtimes: List[RuntimeProvider], **kwargs
    ) -> None:
        runtime_results = []
        runtimes = onnx_runtimes if self.target_format == Format.ONNX else format2runtimes(self.target_format)
        for runtime_provider in runtimes:
            err_msg = {}
            if self.err_msg:
                err_msg[self.command_type.value] = self.err_msg

            runtime_results.append(
                RuntimeResults(
                    runtime=runtime_provider,
                    status=self.status,
                    tolerance=None,
                    err_msg=err_msg,
                )
            )

        package_descriptor.navigator_status.model_status.append(
            ModelStatus(
                format=self.target_format,
                path=self.output,
                torch_jit=self.target_jit_type,
                precision=self.target_precision,
                runtime_results=runtime_results,
            )
        )
