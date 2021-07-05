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

# Overview

The Triton Model Navigator provides a [run](run.md) command to perform the step-by-step process of:
- [Conversion](converion.md) - converts input model to the formats optimized for inference
- [Triton Model Configuration](triton_model_configurator.md) - creates a model entry in the Triton Model Repository, including framework-specific backend accelerations
- [Profiling](profiling.md) - profiles model's performance using [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer)
- [Analysis](analysis.md):
    - analyzes models using [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer)
    - selects top N configurations within given constraints and objectives
    - generates a summary report for selected configurations
- [Helm Charts](helm_charts.md) - generates Helm Charts for selected configurations

The Triton Model Navigator allows running processes as a single command or performs step execution running each command separately.
