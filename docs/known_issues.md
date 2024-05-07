<!--
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

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
# Known Issues and Limitations

- nav.Module moves original torch.nn.Module to the CPU, in case of weight sharing that might result in unexpected behavior
- For data dependent dynamic control flow (multiple computation graphs) nav.Module might copy the weights for each separate graph
- Source model running in Python can cause OOM issue when GPU memory is larger than CPU RAM memory
- Verify command could potentially experience CUDA OOM errors while trying to run inference on two models at the same time.
- Dependencies between modules in optimized pipelines may lead to unexpected behavior and failure in Inplace Optimize
- TensorRT might require manual installation of correct version of `nvidia-cudnn-cu12` package
- ONNXRuntime 1.17.x does not support ONNX IR 10 (onnx ver 1.16.0)
- ONNXRuntime 1.17.x requires cuDNN 8.x
- DistillBERT ONNX dynamo export does not support dynamic shapes