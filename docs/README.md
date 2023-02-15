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

# Triton Model Navigator

The [Triton Model Navigator](https://github.com/triton-inference-server/model_navigator) automates
the process of moving model from source to deployment on NVIDIA Triton Inference Server. The tool validate possible
export and conversion paths to serializable formats like NVIDIA TensorRT and select the most promising format for
production deployment.

# How it works?

The Triton Model Navigator is designed to provide a single entrypoint for each supported framework. The usage is
simple as call to dedicated `optimize` function to start the process of searching for the best
possible deployment by going through a broad spectrum of model conversions.

The `optimize` internally performs model export, conversion, correctness testing, performance profiling,
and saves all generated artifacts in the `navigator_workspace`, which is represented by a returned `package` object.
The result of `optimize` process can be saved as a portable Navigator Package with the `save` function.
Saved packages only contain the base model formats along with the best selected format based on latency and throughput.
The package can be reused to recreate the process on same or different hardware. The configuration and execution status
is saved in the `status.yaml` file located inside the workspace and the `Navigator Package`.

Finally, the `Navigator Packge` can be used for model deployment
on [Triton Inference Server](https://github.com/triton-inference-server/server). Dedicated API helps with obtaining all
necessary parameters and creating `model_repository` or receive the optimized model for inference in Python environment.

