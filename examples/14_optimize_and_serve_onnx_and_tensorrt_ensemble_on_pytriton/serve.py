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
"""PyTrtion server for ONNX and TensorRT models ensemble."""

import pathlib

from pytriton.decorators import batch  # pytype: disable=import-error
from pytriton.triton import Triton  # pytype: disable=import-error

import model_navigator as nav
from model_navigator.configuration import TensorType


def main():
    """Load package and serve it on PyTriton"""

    onnx_package = nav.package.load("onnx_linear.nav", pathlib.Path("onnx_load_workspace"))
    tensort_package = nav.package.load("tensorrt_linear.nav", pathlib.Path("tensorrt_load_workspace"))
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

    """Use TensorType.TORCH to return torch.Tensor from runner inference function and enable zero-copy inference."""
    onnx_pytriton_adapter = nav.pytriton.PyTritonAdapter(
        package=onnx_package, strategy=nav.MaxThroughputStrategy(), runner_return_type=TensorType.TORCH
    )
    onnx_runner = onnx_pytriton_adapter.runner
    onnx_runner.activate()

    tensorrt_pytriton_adapter = nav.pytriton.PyTritonAdapter(
        package=tensort_package, strategy=nav.MaxThroughputStrategy()
    )
    tensorrt_runner = tensorrt_pytriton_adapter.runner
    tensorrt_runner.activate()

    @batch
    def infer_func(**inputs):
        """Wrap runner inference function and add `@batch` decorator to enable batching."""

        onnx_output = onnx_runner.infer(inputs)

        tensorrt_input = {"input__0": onnx_output["output__0"]}

        tensorrt_output = tensorrt_runner.infer(tensorrt_input)

        return tensorrt_output

    """Connecting inference callback with Triton Inference Server."""
    with Triton() as triton:
        """Load model into Triton Inference Server."""
        triton.bind(
            model_name="linear",
            infer_func=infer_func,
            inputs=onnx_pytriton_adapter.inputs,
            outputs=onnx_pytriton_adapter.outputs,
            config=onnx_pytriton_adapter.config,
        )
        """Serve model through Triton Inference Server."""
        triton.serve()


if __name__ == "__main__":
    main()
