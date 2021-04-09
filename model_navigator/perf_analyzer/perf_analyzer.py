# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from typing import List

import logging
from subprocess import (
    PIPE,
    STDOUT,
    CalledProcessError,
    Popen,
    TimeoutExpired,
    check_output,
)

from ..model_navigator_exceptions import ModelNavigatorException

MAX_INTERVAL_CHANGES = 10
INTERVAL_DELTA = 1000

LOGGER = logging.getLogger(__name__)


class PerfAnalyzer:
    """
    This class provides an interface for running workloads
    with perf_analyzer.
    """

    def __init__(self, config, stream_output: bool = False):
        """
        Parameters
        ----------
        path : full path to the perf_analyzer
                executable
        config : PerfAnalyzerConfig
            keys are names of arguments to perf_analyzer,
            values are their values.
        """
        self.bin_path = "perf_analyzer"
        self._config = config
        self._output = None
        self._stream_output = stream_output

    def run(self):
        """
        Runs the perf analyzer with the
        intialized configuration

        Returns
        -------
        List of Records
            List of the metrics obtained from this
            run of perf_analyzer

        Raises
        ------
        ServicAnalyzerException
            If subprocess throws CalledProcessError
        """
        if self._stream_output:
            self._output = str()

        for _ in range(MAX_INTERVAL_CHANGES):
            cmd = [self.bin_path]
            cmd += self._config.to_cli_string().replace("=", " ").split()

            timeout = self._config["measurement-interval"] * 10 / 1000
            LOGGER.debug(f"Perf Analyze command: {cmd}")
            LOGGER.debug(f"Perf Analyze command timeout: {timeout}s")
            try:
                if self._stream_output:
                    self._run_with_stream(cmd=cmd, timeout=timeout)
                else:
                    self._output = check_output(
                        cmd,
                        start_new_session=True,
                        stderr=STDOUT,
                        encoding="utf-8",
                        timeout=timeout,
                    )
                return
            except CalledProcessError as e:
                if e.output.find("Please use a larger time window.") > 0:
                    self._config["measurement-interval"] += INTERVAL_DELTA
                    LOGGER.info(
                        "perf_analyzer's measurement window is too small, "
                        f"increased to {self._config['measurement-interval']} ms."
                    )
                else:
                    raise ModelNavigatorException(
                        f"Running perf_analyzer with {e.cmd} failed with" f" exit status {e.returncode} : {e.output}"
                    )
            except TimeoutExpired:
                self._config["measurement-interval"] += INTERVAL_DELTA
                LOGGER.info("perf_analyzer's timeouted, " f"increased to {self._config['measurement-interval']} ms.")

        raise ModelNavigatorException(
            f"Ran perf_analyzer {MAX_INTERVAL_CHANGES} times, "
            "but no valid requests recorded in max time interval"
            f" of {self._config['measurement-interval']} "
        )

    def output(self):
        """
        Returns
        -------
        The stdout output of the
        last perf_analyzer run
        """
        if self._output:
            return self._output
        raise ModelNavigatorException("Attempted to get perf_analyzer output" "without calling run first.")

    def _run_with_stream(self, cmd: List[str], timeout: int):
        command = ["timeout", str(timeout)]
        command.extend(cmd)
        process = Popen(command, start_new_session=True, stdout=PIPE, encoding="utf-8")
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                self._output += output
                print(output.rstrip())

        result = process.poll()
        return result
