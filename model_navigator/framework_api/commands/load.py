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

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy
import yaml

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.status import NavigatorStatus
from model_navigator.framework_api.utils import get_default_status_filename, get_package_path

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


class LoadMetadata(Command):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(name="Load metadata.", command_type=CommandType.CUSTOM, requires=requires)

    @staticmethod
    def get_output_name():
        return "input_metadata", "output_metadata"

    def _update_package_descriptor(self, package_descriptor: "PackageDescriptor", **kwargs) -> None:
        (
            package_descriptor.navigator_status.input_metadata,
            package_descriptor.navigator_status.output_metadata,
        ) = self.output

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        **kwargs,
    ) -> Tuple[TensorMetadata, TensorMetadata]:

        package_path = get_package_path(workdir, model_name)
        with open(package_path / get_default_status_filename()) as f:
            navigator_status = NavigatorStatus.from_dict(yaml.safe_load(f))

        return navigator_status.input_metadata, navigator_status.output_metadata


class LoadSamples(Command):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(name="Load samples.", command_type=CommandType.CUSTOM, requires=requires)

    @staticmethod
    def get_output_name():
        return (
            "profiling_sample",
            "correctness_samples",
            "conversion_samples",
            "profiling_sample_output",
            "correctness_samples_output",
            "conversion_samples_output",
        )

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        batch_dim: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorMetadata, TensorMetadata]:

        package_path = get_package_path(workdir, model_name)
        ret = []
        for samples_name in self.get_output_name():
            samples_type = samples_name.split("_")[0]
            samples_dirname = "model_output" if samples_name.split("_")[-1] == "output" else "model_input"
            samples_dirpath = package_path / samples_dirname / samples_type
            samples = []
            for sample_filepath in samples_dirpath.iterdir():
                sample = {}
                with numpy.load(sample_filepath.as_posix()) as data:
                    for k, v in data.items():
                        if batch_dim is not None:
                            v = numpy.expand_dims(v, batch_dim)
                            # v = numpy.repeat(v, max_batch_size, batch_dim)
                        sample[k] = v
                samples.append(sample)
            if samples_type == "profiling":
                samples = samples[0]
            ret.append(samples)

        return tuple(ret)
