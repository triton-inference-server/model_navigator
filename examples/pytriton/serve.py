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

from pytriton.decorators import batch  # pytype: disable=import-error
from pytriton.triton import Triton  # pytype: disable=import-error

import model_navigator as nav

package = nav.package.load("mlp.nav", "load_workspace")

pytriton_adapter = nav.pytriton.PyTritonAdapter(package=package)
runner = pytriton_adapter.runner
runner.activate()


@batch
def infer_func(**inputs):
    return runner.infer(inputs)


# Connecting inference callback with Triton Inference Server
with Triton() as triton:
    # Load model into Triton Inference Server
    triton.bind(
        model_name="mlp",
        infer_func=infer_func,
        inputs=pytriton_adapter.inputs,
        outputs=pytriton_adapter.outputs,
        config=pytriton_adapter.config,
    )
    # Serve model through Triton Inference Server
    triton.serve()
