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

# Quick Start

Using Model Navigator is as simply as calling `optimize` with `model` and `dataloader`:
The `optimize` function will save all the artifacts it generates in the `navigator_workspace`.

**Note:** The `dataloader` is utilized to determine the maximum and minimum shapes of the inputs utilized during model conversions. The `Model Navigator` employs a single sample from the `dataloader`, which is then repeated to generate synthetic batches for profiling purposes. Correctness tests are conducted on a subset of the `dataloader` samples, while verification tests are executed on the entire `dataloader`.

```python
import torch
import model_navigator as nav

nav.torch.optimize(
    model=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True).eval(),
    dataloader=[torch.randn(1, 3, 256, 256) for _ in range(10)],
)
```

The code snippet below demonstrates the usage of the `PyTritonAdapter` to retrieve the `runner` and other necessary information. The `runner` serves as an abstraction that connects the model checkpoint with its runtime, making the inference process more accessible and straightforward. Following that, it initiates the [PyTriton](https://github.com/triton-inference-server/pytriton) server using the provided parameters.

```python
pytriton_adapter = nav.pytriton.PyTritonAdapter(package=package, strategy=nav.MaxThroughputStrategy())
runner = pytriton_adapter.runner

runner.activate()

@batch
def infer_func(**inputs):
    return runner.infer(inputs)

with Triton() as triton:
    triton.bind(
        model_name="resnet50",
        infer_func=infer_func,
        inputs=pytriton_adapter.inputs,
        outputs=pytriton_adapter.outputs,
        config=pytriton_adapter.config,
    )
    triton.serve()
```

Alternatively, Model Navigator can generate `model_repository` that can be served on the [Triton Inference Server](https://github.com/triton-inference-server/server):

```python
nav.triton.model_repository.add_model_from_package(
    model_repository_path=pathlib.Path("model_repository"),
    model_name="resnet50",
    package=package,
    strategy=nav.MaxThroughputStrategy(),
)
```

For more information on additional frameworks and `optimize` function parameters, please refer to the API documentation and examples.
