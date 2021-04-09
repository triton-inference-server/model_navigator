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
from collections import deque

import sh
from model_navigator import Format
from model_navigator.log import section_header
from model_navigator.resources import FORMAT2RESOURCE, FORMAT_RESOURCE

LOGGER = logging.getLogger(__name__)
MODEL_TRANSFORM_SEP = "."
TRANSFORM_SPEC_SEP = "-"
PARAMETERS_SEP = "_"
KEY_VALUE_SEP = ""


def extend_model_name(model_name, transform_name):
    # extend transform spec with current transform name
    basename, *transform_spec = model_name.split(MODEL_TRANSFORM_SEP)
    prev_transform_names = transform_spec[0].split(TRANSFORM_SPEC_SEP) if transform_spec else []
    transform_spec = TRANSFORM_SPEC_SEP.join(prev_transform_names + [transform_name])
    return f"{basename}{MODEL_TRANSFORM_SEP}{transform_spec}"


def execute_sh_command(cmd: sh.Command, *, log_file, verbose: bool = False):
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


def prepare_log_header(file_handler, src_format: Format, dst_format: Format):
    lines = list()
    lines.append(
        f"Performing optimization from format {FORMAT2RESOURCE[src_format]} to {FORMAT2RESOURCE[dst_format]}.\n"
    )
    lines.append("In case of any issue please review helpful link section to address problems correctly.")

    lines.append(section_header("Helpful links"))

    src_resource = FORMAT_RESOURCE[src_format]
    dst_resource = FORMAT_RESOURCE[dst_format]
    resources = {src_resource, dst_resource}
    for resource in resources:
        lines.append(f"{resource.name}: {resource.link}")

    header = "\n".join(lines)
    header += "\n"
    header += section_header("Optimization Log")

    file_handler.write(header)
