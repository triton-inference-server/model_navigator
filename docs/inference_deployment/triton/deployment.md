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
# Deployment on Triton Inference Server

The Triton Model Navigator provides an API for working with the Triton model repository. Currently, we
support adding your model or a pre-selected model from a Navigator Package.

The API only provides possible functionality for the given model's type and only provides offline validation of
the provided configuration. In the end, the model with the configuration is created inside the provided model
repository path.

## Adding your model to the Triton model repository

When you work with an already exported model, you can provide a path to where one's model is located.
Then you can use one of the specialized APIs that guides you through what options are possible for deployment of the
selected model type.

Example of deploying a TensorRT model:

```python
import model_navigator as nav

nav.triton.model_repository.add_model(
    model_repository_path="/path/to/triton/model/repository",
    model_path="/path/to/model/plan/file",
    model_name="NameOfModel",
    config=nav.triton.TensorRTModelConfig(
        max_batch_size=256,
        optimization=nav.triton.CUDAGraphOptimization(),
        response_cache=True,
    )
)
```

The model catalog with the model file and configuration will be created inside `model_repository_path`. More
about the function you can find in the [adding model section](api/adding_model.md).

## Adding model from package to the Triton model repository

When you want to deploy a model from a package created during the `optimize` process, you can use:

```python
import model_navigator as nav

nav.triton.model_repository.add_model_from_package(
    model_repository_path="/path/to/triton/model/repository",
    model_name="NameOfModel",
    package=package,
)
```

The model is automatically selected based on profiling results. The default selection options can be adjusted by
changing the `strategy` argument. More
about the function you can find in [adding model section](api/adding_model.md).

## Using Triton Model Analyzer

A model added to the Triton Inference Server can be further optimized in the target environment
using [the Triton Model Analyzer](https://pypi.org/project/triton-model-analyzer/).

Please follow the [documentation](https://github.com/triton-inference-server/model_analyzer) to learn more about how
to use the Triton Model Analyzer.
