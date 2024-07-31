# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
"""PyTrtion server for linear model"""

from pytriton.decorators import batch  # pytype: disable=import-error
from pytriton.triton import Triton  # pytype: disable=import-error

import model_navigator as nav


def main():
    """Load package and serve it on PyTriton"""

    package = nav.package.load("linear.nav", "load_workspace")

    """
    Create a PyTritonAdapter instance for given package and strategy.

    PyTritonAdapter is a wrapper around the package,
    that provides a unified interface for PyTriton integration.

    Wrapper provided all necessary information for inference:
    - inputs_metadata
    - outputs_metadata
    - PyTriton configuration
    - Runner selected with given strategy (defaults to MaxThroughputAndMinLatencyStrategy)
    """
    pytriton_adapter = nav.pytriton.PyTritonAdapter(package=package)
    runner = pytriton_adapter.runner

    """Runner must be activated before inference."""
    runner.activate()

    @batch
    def infer_func(**inputs):
        """Wrap runner inference function and add `@batch` decorator to enable batching."""

        return runner.infer(inputs)

    """Connecting inference callback with Triton Inference Server."""
    with Triton() as triton:
        """Load model into Triton Inference Server."""
        triton.bind(
            model_name="linear",
            infer_func=infer_func,
            inputs=pytriton_adapter.inputs,
            outputs=pytriton_adapter.outputs,
            config=pytriton_adapter.config,
        )
        """Serve model through Triton Inference Server."""
        triton.serve()


if __name__ == "__main__":
    main()
