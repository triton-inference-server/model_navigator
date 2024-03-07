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

# Optimize model and use zero-copy runners

In this example we show how to Model Navigator Inplace Optimize to run optimized models in place of the PyTorch models in the original pipeline.
Depending on the mode the `optimize.py` script can run PyTorch Stable Diffusion pipeline or optimize and run the CLIP, U-Net and VAE models in TensorRT without any changes to the original pipeline.

We recommend running this example in NVIDIA NGC [PyTorch containter](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) with CUDA 12 and TensorRT >= 8.6.
The Python script `optimize.py` wraps the Python model using Inplace Optimize and then runs profiling.

To run the optimization and profiling run the script:

```bash
./optimize.py
```