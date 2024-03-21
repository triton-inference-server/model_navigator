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
"""Correctness command and it's results."""

import dataclasses
import json
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, Type

from model_navigator.api.config import Format
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.configuration.runner.runner_config import RunnerConfig
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.runners.base import NavigatorRunner
from model_navigator.utils.common import DataObject, parse_kwargs_to_cmd
from model_navigator.utils.format_helpers import is_source_format


@dataclasses.dataclass
class Tolerance(DataObject):
    """Tolerance values."""

    atol: float
    rtol: float

    @classmethod
    def from_dict(cls, tolerance_dict: Dict) -> "Tolerance":
        """Instantiate Tolerance from a dictionary.

        Args:
            tolerance_dict (Dict): Dictionary with tolerance values.

        Returns:
            Tolerance
        """
        return cls(
            atol=tolerance_dict["atol"],
            rtol=tolerance_dict["rtol"],
        )


class TolerancePerOutputName(Dict[str, Tolerance]):
    """Dictionary where key is output name and value is Tolerance."""

    def to_json(self) -> List[Dict]:
        """Parse TolerancePerOutputName to a list of dictionaries.

        Returns:
            List[Dict]
        """
        return [{"output_name": name, **tol.to_dict(parse=True)} for name, tol in self.items()]

    @classmethod
    def from_json(cls, data: List[Dict]) -> "TolerancePerOutputName":
        """Instantiate TolerancePerOutputName from a list of dictionaries.

        Args:
            data (List): TolerancePerOutputName data.

        Returns:
            TolerancePerOutputName
        """
        tol_per_out = cls()
        for tol in data:
            tol_per_out[tol["output_name"]] = Tolerance.from_dict(tol)
        return tol_per_out


class Correctness(Command):
    """Correctness Command."""

    def _run(
        self,
        workspace: Workspace,
        format: Format,
        runner_cls: Type[NavigatorRunner],
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        batch_dim: Optional[int],
        path: pathlib.Path,
        verbose: bool,
        model: Optional[Any] = None,
        runner_config: Optional[RunnerConfig] = None,
    ) -> CommandOutput:
        """Run correctness command.

        Args:
            workspace: Model Navigator workspace path.
            format: Model format.
            runner_cls: Type of a runner to use with a model.
            input_metadata: Input metadata.
            output_metadata: Output metadata.
            batch_dim: Batch dimension.
            path: Model path relative to workspace path.
            verbose: If True verbose logging.
            model: Model if correctness should be run on a source model. Defaults to None.
            runner_config: Additional runner arguments

        Returns:
            CommandOutput: Status OK and TolerancePerOutputName of the model with runner.
        """
        LOGGER.info(f"Correctness test for: {format} {runner_cls} started.")
        model_path = workspace.path / path
        model_dir = model_path.parent

        if not is_source_format(format) and not model_path.exists():
            LOGGER.warning(f"Model: {model_path.as_posix()!r} not found, command skipped.")
            return CommandOutput(status=CommandStatus.SKIPPED)

        with ExecutionContext(
            workspace=workspace,
            script_path=model_dir / f"reproduce_correctness-{runner_cls.slug()}.py",
            cmd_path=model_dir / f"reproduce_correctness-{runner_cls.slug()}.sh",
            verbose=verbose,
        ) as context, tempfile.NamedTemporaryFile() as temp_file:
            kwargs = {
                "navigator_workspace": workspace.path.as_posix(),
                "batch_dim": batch_dim,
                "results_path": temp_file.name,
                "runner_name": runner_cls.name(),
                "input_metadata": input_metadata.to_json(),
                "output_metadata": output_metadata.to_json(),
                "runner_config": runner_config.to_dict(parse=True) if runner_config else None,
            }

            from model_navigator.commands.correctness import correctness_script

            if is_source_format(format):
                correctness_script.get_model = lambda: model
                args = parse_kwargs_to_cmd(kwargs)
                context.execute_local_runtime_script(correctness_script.__file__, correctness_script.correctness, args)
            else:
                kwargs["model_path"] = path
                args = parse_kwargs_to_cmd(kwargs)
                context.execute_external_runtime_script(correctness_script.__file__, args)
            per_output_tolerance = TolerancePerOutputName.from_json(json.load(temp_file))

        return CommandOutput(status=CommandStatus.OK, output={"per_output_tolerance": per_output_tolerance})
