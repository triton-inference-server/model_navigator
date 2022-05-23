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
import contextlib
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import deque

import Pyro4.util
import Pyro4.utils.flame

from model_navigator.framework import PyTorch, TensorFlow2
from model_navigator.model import Format
from model_navigator.utils.resources import FORMAT2RESOURCE, FORMAT_RESOURCE

LOGGER = logging.getLogger(__name__)
MODEL_COMMAND_SEP = "."
COMMAND_SPEC_SEP = "-"
PARAMETERS_SEP = "_"
KEY_VALUE_SEP = ""

FORMAT2FRAMEWORK = {
    Format.TORCHSCRIPT: PyTorch,
    Format.ONNX: PyTorch,
    Format.TENSORRT: PyTorch,
    Format.TF_SAVEDMODEL: TensorFlow2,
}


def extend_model_name(model_name, transform_name):
    # extend transform spec with current transform name
    basename, *transform_spec = model_name.split(MODEL_COMMAND_SEP)
    prev_transform_names = transform_spec[0].split(COMMAND_SPEC_SEP) if transform_spec else []
    transform_spec = COMMAND_SPEC_SEP.join(prev_transform_names + [transform_name])
    return f"{basename}{MODEL_COMMAND_SEP}{transform_spec}"


def execute_sh_command(cmd, *, log_file, verbose: bool = False):
    tf_cpp_min_log_level = 0 if verbose else 2
    envs = {
        "TF_CPP_MIN_LOG_LEVEL": str(tf_cpp_min_log_level),
        "TF_ENABLE_DEPRECATION_WARNINGS": "0",
    }
    LOG_TAIL_SIZE = 32  # number of lines to show on error
    log_buffer = deque(maxlen=LOG_TAIL_SIZE)

    cmd_str = str(cmd)
    sep = "=" * 32 + "\n"
    cmd_str = sep + cmd_str + "\n" + sep + "\n"

    if verbose:
        print(cmd_str)
    log_file.writelines([cmd_str])
    for line in cmd(_iter=True, _err_to_out=True, _out=log_file, _env=envs, _bg_exc=False, _tee=True):
        line = line.rstrip()
        if verbose:
            print("\t" + line, flush=True)
        else:
            log_buffer.append(line)


def _handle_subprocess_output(p, log_file, verbose):
    for line in p.stdout:
        if log_file:
            log_file.write(line)
        line = line.rstrip()
        if verbose:
            print("\t" + line, flush=True)


@contextlib.contextmanager
def navigator_subprocess(*, log_file=None, verbose: bool = False):
    tf_cpp_min_log_level = 0 if verbose else 2
    envs = {
        "TF_CPP_MIN_LOG_LEVEL": str(tf_cpp_min_log_level),
        "TF_ENABLE_DEPRECATION_WARNINGS": "0",
        "PYRO_FLAME_ENABLED": "1",
    }

    msg = None
    Pyro4.config.SERIALIZER = "pickle"  # pytype: disable=module-attr
    socket = f"/tmp/model-navigator-{secrets.token_urlsafe()}.socket"
    with subprocess.Popen(
        [sys.executable, "-m", "Pyro4.utils.flameserver", "-u", socket, "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=envs,
        encoding="utf-8",
    ) as p:
        t = threading.Thread(target=_handle_subprocess_output, args=(p, log_file, verbose))
        t.start()
        try:
            while not os.path.exists(socket) and p.returncode is None:
                p.poll()
                time.sleep(0.01)
            with Pyro4.utils.flame.connect(f"./u:{socket}") as flame:
                yield flame
        except Exception:
            tb = "".join(Pyro4.util.getPyroTraceback())
            msg = f"Uncaught exception:\n {tb}\n"
            if verbose:
                for line in msg.splitlines():
                    print("\t" + line, flush=True)
            raise
        finally:
            p.terminate()
            t.join()
            if msg is not None and log_file is not None:
                log_file.write(msg)


def prepare_log_header(file_handler, src_format: Format, dst_format: Format):
    lines = [
        f"Performing optimization from format {FORMAT2RESOURCE[src_format]} to {FORMAT2RESOURCE[dst_format]}.\n",
        "There should be available JSON files with inputs and outputs used during verification "
        "in case of failed conversion.",
        "Refer to the [Verification of Conversion Correctness]"
        "(https://github.com/triton-inference-server/model_navigator/blob/main/docs/conversion.md#verification-of-conversion-correctness) "
        "section of conversion command documentation for details.\n",
        "In case of any issue please review helpful link section to address problems correctly.",
        _section_header("Helpful links"),
    ]

    src_resource = FORMAT_RESOURCE[src_format]
    dst_resource = FORMAT_RESOURCE[dst_format]
    resources = {src_resource, dst_resource}
    for resource in resources:
        lines.append(f"{resource.name}: {resource.link}")

    header = "\n".join(lines)
    header += "\n"
    header += _section_header("Optimization Log")

    file_handler.write(header)


def _section_header(section_name: str) -> str:
    return f"\n\n================== {section_name} ==================\n\n"
