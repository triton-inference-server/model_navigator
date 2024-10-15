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
"""Base class for profile report classes."""

import enum
from dataclasses import dataclass, field
from typing import List, Optional

from pyee import EventEmitter
from rich.console import Console
from rich.table import Table
from rich.text import Text

from model_navigator.inplace.profiling import ProfilingResult
from model_navigator.reporting.profile.events import ProfileEvent, default_event_emitter


class OpStatus(enum.Enum):
    """Enumerate of available op statuses."""

    OK = "OK"
    FAILED = "FAILED"


@dataclass
class Row:
    """Represents a row in the status table."""

    runtime: str = ""
    status: Text = field(default_factory=Text)
    results: List[ProfilingResult] = field(default_factory=list)
    is_separator: bool = False


class SimpleReport:
    """Profile reports."""

    def __init__(self, show_results: bool = True, event_emitter: Optional[EventEmitter] = None) -> None:
        """Initialized object.

        Args:
            show_results: Whether to show results or not.
            event_emitter: optional emitter to register for events.
            width: Width for the displayed line.
        """
        self.emitter = event_emitter or default_event_emitter()
        self.errored_runtimes = []
        self.profiling_results = {}
        self.current_runtime = None
        self.current_sample_id = None
        self.profiling_started = False
        self.show_results = show_results

        self.status_line = None

        self.table_data: List[Row] = []

        self.console = Console(record=True, width=256)  # specify width to prevent auto-width detection
        self.listen_for_events()

    def listen_for_events(self):
        """Register listener on events."""
        self.emitter.on(ProfileEvent.PROFILING_STARTED, self.on_profiling_started)
        self.emitter.on(ProfileEvent.PROFILING_FINISHED, self.on_profiling_finished)

        self.emitter.on(ProfileEvent.RUNTIME_PROFILING_STARTED, self.on_runtime_profiling_started)
        self.emitter.on(ProfileEvent.RUNTIME_PROFILING_FINISHED, self.on_runtime_profiling_finished)
        self.emitter.on(ProfileEvent.RUNTIME_PROFILING_ERROR, self.on_runtime_profiling_error)

        self.emitter.on(ProfileEvent.RUNTIME_PROFILING_RESULT, self.on_runtime_profiling_result)

    def on_profiling_started(
        self,
    ):
        """Action on profiling started event."""
        self.profiling_started = True
        self.console.print("[bold]Profiling started[/bold]")

    def on_profiling_finished(self):
        """Action on profiling finished event."""
        self.profiling_started = False
        self.console.print("[bold]Profiling finished[/bold]")
        self.generate_table()

    def on_runtime_profiling_started(self, name: str):
        """Action on runtime profiling started event."""
        self.current_runtime = name
        self.status_line = f"[bold]{self.current_runtime}:[/bold] profiling"
        self.console.print(f"{self.status_line} ...")
        self.profiling_results[self.current_runtime] = []

    def on_runtime_profiling_finished(self):
        """Action on runtime profiling finished event."""
        self.table_data.append(
            Row(
                runtime=self.current_runtime,
                status=Text(OpStatus.OK.value, style="green"),
                results=self.profiling_results[self.current_runtime],
            )
        )
        self.table_data.append(Row(is_separator=True))
        self.print_status_line(status=OpStatus.OK)

        self.status_line = None
        self.current_runtime = None

    def on_runtime_profiling_error(self):
        """Action on runtime profiling finished event."""
        self.table_data.append(Row(runtime=self.current_runtime, status=Text(OpStatus.FAILED.value, style="red")))
        self.table_data.append(Row(is_separator=True))

        self.print_status_line(status=OpStatus.FAILED)

        self.status_line = None
        self.current_runtime = None

    def on_runtime_profiling_result(self, result: ProfilingResult):
        """Action on sample profiling finished event."""
        self.profiling_results[self.current_runtime].append(result)
        if self.show_results:
            result_line = (
                f"""[bold]{self.current_runtime}:[/bold] """
                f"""Batch {result.batch_size:6}, """
                rf"""Throughput: {result.throughput:10.2f} \[infer/sec], """
                rf"""Avg Latency: {result.avg_latency:10.2f} \[ms]"""
            )
            self.console.print(result_line)

    def print_status_line(self, status: OpStatus):
        """Print console log on operation status."""
        if self.status_line is not None:
            if status == OpStatus.OK:
                style = "green"
            elif status == OpStatus.FAILED:
                style = "red"
            else:
                style = "black"
            self.console.print(f"{self.status_line} [bold {style}]{status.value}[/bold {style}]")

    def generate_table(self):
        """Generates a status table based on the current table data."""
        table = Table(title="Profiling status")
        table.add_column("Model on Runtime", style="cyan", header_style="bold cyan")
        table.add_column("Status", justify="center")
        table.add_column("Batch", style="magenta", justify="left", header_style="bold blue")
        table.add_column("Throughput [infer/sec]", style="blue", justify="left", header_style="bold blue")
        table.add_column("Avg Latency [ms]", style="blue", justify="left", header_style="bold blue")

        for row in self.table_data:
            if row.is_separator:
                table.add_section()
            else:
                batch = []
                throughput = []
                latency = []
                for result in row.results:
                    batch.append(f"{result.batch_size:6}")
                    throughput.append(f"{result.throughput:10.2f}")
                    latency.append(f"{result.avg_latency:10.2f}")

                table.add_row(
                    row.runtime,
                    row.status,
                    "\n".join(batch),
                    "\n".join(throughput),
                    "\n".join(latency),
                )

        self.console.print(table)
