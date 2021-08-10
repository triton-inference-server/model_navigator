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
import logging
import os
import pty
import select
import shlex
import subprocess
import sys
import termios
import tty
from typing import List

LOGGER = logging.getLogger(__name__)


def execute_in_subprocess(cmd: List[str], **kwargs):
    """
    Execute a process and stream output to logger
    :param cmd: command and arguments to run
    :type cmd: List[str]
    """

    # function copied from gh:apache/airflow project

    output = b""

    LOGGER.info("Executing cmd: %s", " ".join(shlex.quote(c) for c in cmd))
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, close_fds=True, **kwargs
    ) as proc:
        if proc.stdout:
            with proc.stdout:
                for line in iter(proc.stdout.readline, b""):
                    output += line
        proc.wait()
    return proc, output


def execute_interactive(cmd: List[str], **kwargs):
    """
    Runs the new command as a subprocess and ensures that the terminal's state is restored to its original
    state after the process is completed e.g. if the subprocess hides the cursor, it will be restored after
    the process is completed.
    """

    # function copied from gh:apache/airflow project

    LOGGER.info("Executing cmd: %s", " ".join(shlex.quote(c) for c in cmd))

    output = b""

    old_tty = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

    # open pseudo-terminal to interact with subprocess
    master_fd, slave_fd = pty.openpty()
    try:
        # use os.setsid() make it run in a new process group, or bash job control will not be enabled
        with subprocess.Popen(
            cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd, universal_newlines=True, **kwargs
        ) as proc:
            while proc.poll() is None:
                readable_fbs, _, _ = select.select([sys.stdin, master_fd], [], [], 0.1)
                if sys.stdin in readable_fbs:
                    input_data = os.read(sys.stdin.fileno(), 10240)
                    os.write(master_fd, input_data)
                if master_fd in readable_fbs:
                    output_data = os.read(master_fd, 10240)
                    if output_data:
                        os.write(sys.stdout.fileno(), output_data)
                        output += output_data
    finally:
        # restore tty settings back
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
    return proc, output
