<!--
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

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

- missing support for stateful models (ex. time-series one)
- missing support for models without batching support
- no verification of conversion results for conversions: TF -> ONNX, TorchScript -> ONNX
- issues with TorchScript -> ONNX conversion due to [issue in PyTorch 1.8](https://github.com/pytorch/pytorch/issues/53506)
  - affected NVIDIA PyTorch containers: 20.12, 21.02, 21.03
  - workaround: use PyTorch containers newer than 21.03
- possible to define a single profile for TensorRT
- no support for TensorRT int8 conversions
- no support for TF-TRT conversion
- no custom ops support
- Triton Inference Server stays in the background when the profile
  process is interrupted by the user
