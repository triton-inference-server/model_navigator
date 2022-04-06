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
# Triton Model Navigator

The NVIDIA [Triton Inference Server](https://github.com/triton-inference-server) provides a robust and configurable
solution for deploying and managing AI models.
The [Triton Model Navigator](https://github.com/triton-inference-server/model_navigator) is a tool that provides the
ability to automate the process of moving model from source to optimal format and configuration for deployment on Triton Inference Server.
The tool support export model from source to all possible formats and apply the Triton Inference Server backend optimizations.
Finally, it uses the [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer)
to find the best Triton Model configuration, matches the provided constraints, and optimize performance.

## Export from source

Python Export API that helps with exporting model from framework to all possible formats.
This stage is dedicated to assure the model is inference ready on time of training, by executing conversion, correctness and performance
tests that help to identify model related issues. Artifacts produced by Triton Model Navigator are stored in a `.nav`
package that contains checkpoints and all necessary information for further processing by Model Navigator CLI.

## Optimize for Triton Inference Server

The optimizer part use the generated `.nav` package and run all possible conversion to available formats and apply
addition Triton Inference Server backends optimizations. Finally, it uses internally
the [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer)
to find the best Triton Model configuration, matches the provided constraints, and optimize performance.
