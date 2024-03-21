<!--
Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Optimize Torch linear model and save package

In this example, we show how to save a .nav package. It contains all the information about the performed optimizations and the models themselves. The Navigator package is portable and transferable, and can be reused locally or transferred to another machine.

Only base models and those with minimal latency and maximal throughput are saved.

Base formats are:

* Torch:
    * TorchScript
    * ONNX
* TensorFlow2:
    * SavedModel
* JAX:
    * SavedModel
* ONNX:
    * ONNX

We recommend running this example in NVIDIA NGC [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). To run the example, simply run the `optimize.py` script:

```bash
./optimize.py --output-path linear.nav
```