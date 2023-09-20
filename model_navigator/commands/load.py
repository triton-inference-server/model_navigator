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
"""Commands for loading samples from the drive."""

from typing import Optional

import yaml

from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.core.dataloader import load_samples
from model_navigator.core.workspace import Workspace
from model_navigator.package.status import Status
from model_navigator.utils.common import get_default_status_filename


class LoadMetadata(Command, is_required=True):
    """Load IO metadata from the status.yaml file."""

    def _run(
        self,
        workspace: Workspace,
    ) -> CommandOutput:
        """Run loading IO metadata from the status.yaml.

        Metadata that are being loaded:
            1) Input metadata,
            2) Output metadata,
            3) TensorRT profile,
            4) Maximum batch size from the dataloader.

        Args:
            workspace (Path): Model Navigator workspace path.

        Returns:
            CommandOutput
        """
        with open(workspace.path / get_default_status_filename()) as f:
            status = Status.from_dict(yaml.safe_load(f))
        return CommandOutput(
            status=CommandStatus.OK,
            output={
                "input_metadata": status.input_metadata,
                "output_metadata": status.output_metadata,
                "dataloader_trt_profile": status.dataloader_trt_profile,
                "dataloader_max_batch_size": status.dataloader_max_batch_size,
            },
        )


class LoadSamples(Command):
    """Load IO samples from the drive."""

    def _run(
        self,
        workspace: Workspace,
        batch_dim: Optional[int] = None,
    ) -> CommandOutput:
        """Run loading IO samples.

        Samples that are being loaded:
            1) Profiling samples and it's output,
            2) Conversion samples and their outputs,
            3) Correctness samples and their outputs.

        Args:
            workspace: Model Navigator workspace path.
            batch_dim: Batch dimension. Defaults to None.

        Returns:
            CommandOutput
        """
        samples_name = (
            "profiling_sample",
            "correctness_samples",
            "conversion_samples",
            "profiling_sample_output",
            "correctness_samples_output",
            "conversion_samples_output",
        )
        ret = {}
        for name in samples_name:
            samples = load_samples(name, workspace.path, batch_dim)
            if name.startswith("profiling"):
                ret[name] = samples[0]
            else:
                ret[name] = samples

        return CommandOutput(
            status=CommandStatus.OK,
            output=ret,
        )
