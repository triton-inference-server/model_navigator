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

The NVIDIA [Triton Inference Server](https://github.com/triton-inference-server) provides a robust and configurable solution for deploying and managing AI models. The [Triton
Model Navigator](https://github.com/triton-inference-server/model_navigator) is a tool that provides the ability to automate the process of model deployment on the Triton Inference Server.
The tool optimize models running conversion to available formats and applying addition Triton backends optimizations. Finally, it uses the [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer)
to find the best Triton Model configuration, matches the provided constraints, and optimize performance.

## Documentation

* [Overview](docs/overview.md)
* [Support Matrix](docs/support_matrix.md)
* [Quick Start](docs/quick_start.md)
* [Installation](docs/installation.md)
* [Framework Navigator](docs/framework_navigator.md)
* [Running the Triton Model Navigator](docs/run.md)
* [Model Conversions](docs/conversion.md)
* [Triton Model Configurator](docs/triton_model_configurator.md)
* [Models Profiling](docs/profiling.md)
* [Models Analysis](docs/analysis.md)
* [Helm Charts](docs/helm_charts.md)
* [Changelog](CHANGELOG.md)
* [Known Issues](docs/known_issues.md)
* [Contributing](CONTRIBUTING.md)
