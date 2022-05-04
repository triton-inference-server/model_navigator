<!--
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

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

- missing support for stateful models (e.g. time-series)
- no verification of conversion results for conversions: TF -> ONNX, TF->TF-TRT, TorchScript -> ONNX
- only possible to define a single profile for TensorRT
- no custom ops support
- Triton Inference Server stays in the background when the profile
  process is interrupted by the user
- when using advanced mode, pytorch model outputs have to be specified in their positional order
