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


from typing import Optional

import fire
import torch  # pytype: disable=import-error

from model_navigator.framework_api.utils import JitType, load_samples


def get_model():
    raise NotImplementedError("Please implement the get_model() function to reproduce the export error.")


def export(
    exported_model_path: str, target_jit_type: str, package_path: str, batch_dim: Optional[int], target_device: str
):
    model = get_model()
    target_jit_type = JitType(target_jit_type)
    profiling_sample = load_samples("profiling_sample", package_path, batch_dim)

    if target_jit_type == JitType.SCRIPT:
        script_module = torch.jit.script(model)
    else:
        dummy_input = tuple(torch.from_numpy(val).to(target_device) for val in profiling_sample.values())
        script_module = torch.jit.trace(model, dummy_input)

    torch.jit.save(script_module, exported_model_path)


if __name__ == "__main__":
    fire.Fire(export)
