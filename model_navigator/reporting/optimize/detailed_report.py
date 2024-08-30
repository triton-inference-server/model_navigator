# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Detailed report for the optimization."""

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional

from pyee import EventEmitter
from rich.table import Table
from rich.text import Text

from model_navigator.commands.base import CommandStatus
from model_navigator.reporting.optimize.simple_report import SimpleReport


@dataclass
class Row:
    """Represents a row in the status table."""

    pipeline: str = ""
    command: str = ""
    config: str = ""
    runner: str = ""
    duration: str = ""
    status: str = ""
    is_separator: bool = False


class DetailedReport(SimpleReport):
    """Detailed Report - similar to simple one but adds status tables after optimization."""

    def __init__(self, event_emitter: Optional[EventEmitter] = None) -> None:
        """Initialized object.

        Args:
            event_emitter: optional emitter to register for events.
        """
        super().__init__(event_emitter)
        self.command_start_time = 0
        self.table_data: List[Row] = []

    def on_optimization_finished(self):
        """Action on optimization finished event."""
        # first generate table
        self.generate_table()
        self.table_data = []
        # then parent final report (if not inplace)
        super().on_optimization_finished()

    def on_pipeline_finished(self):
        """Action on pipeline finished event."""
        super().on_pipeline_finished()
        self.table_data.append(Row(is_separator=True))

    def on_command_started(self, command: str, config_key: Optional[str], runner_name: Optional[str]):
        """Action on command started event."""
        super().on_command_started(command, config_key, runner_name)
        # print pipeline name only for first command
        pipeline = self.current_pipeline if self.is_first_pipeline_command else ""
        self.command_start_time = perf_counter()
        self.table_data.append(
            Row(
                pipeline=pipeline,
                command=command,
                config=config_key or "",
                runner=runner_name or "",
                status="",
            )
        )

    def on_command_finished(self, status: str):
        """Action on command finished event."""
        super().on_command_finished(status)
        # update last status
        command_duration = perf_counter() - self.command_start_time
        status = CommandStatus(status)
        style = None
        if status == CommandStatus.OK:
            style = "green"
        elif status == CommandStatus.FAIL:
            style = "red"
        last_row = self.table_data[-1]
        last_row.status = Text(f"{status}", style=style)
        last_row.duration = f"{command_duration:.1f}s"
        self.is_first_pipeline_command = False

    def generate_table(self):
        """Generates a status table based on the current table data."""
        table = Table(
            title=f"Optimization status for {self.current_module_name}"
            if self.current_module_name
            else "Optimization status"
        )
        table.add_column("Pipeline", style="cyan", header_style="bold cyan")
        table.add_column("Command", style="magenta", header_style="bold magenta")
        table.add_column("Config", style="green", header_style="bold green")
        table.add_column("Runner", style="blue", header_style="bold blue")
        table.add_column("Duration", justify="right", style="red", header_style="bold red")
        table.add_column("Status", justify="center")

        for row in self.table_data:
            if row.is_separator:
                table.add_section()
            else:
                table.add_row(
                    row.pipeline,
                    row.command,
                    row.config,
                    row.runner,
                    row.duration,
                    row.status,
                )

        self.console.print(table)
