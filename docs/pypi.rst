..
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

Triton Model Navigator
========================

Model conversion plays a crucial role in unlocking the maximum performance capabilities of the underlying hardware.
By applying these transformation techniques, models can be optimized to fully utilize the specific features and optimizations offered by the hardware architecture.
Furthermore, conversions allow for serialization of models, separating them from the source code.
This serialization process enhances portability, allowing the models to be seamlessly deployed in production environments.
The decoupling of models from the source code also facilitates maintenance, updates, and collaboration among developers.

However, this process comprises multiple steps and offers various potential paths,
making manual execution complicated and time-consuming. The `Triton Model Navigator`_ offers a user-friendly and automatic solution for converting,
optimizing and deploying machine learning models. It offers a single entry point for various supported frameworks,
allowing users to start the process of searching for the best deployment option with a single call to the dedicated `optimize` function.
Model Navigator handles model export, conversion, correctness testing, and performance profiling, saving all generated artifacts.

Throughout the optimize process, the Model Navigator considers multiple factors, including different precisions,
runtimes, and data shapes. This careful consideration allows for the adjustment of the model's configuration to achieve improved performance,
effectively minimizing the costs associated with serving the model.

Converted models can be easily deployed on the `PyTriton`_ or `Triton Inference Server`_ .


The Model Navigator generates multiple optimized and production-ready models.
The table below illustrates the model formats that can be obtained by using the Model Navigator with various frameworks.

**Table:** Supported conversion target formats per each supported Python framework or file.

+--------------------+--------------------+--------------------+----------+
| PyTorch            | TensorFlow 2       | JAX                | ONNX     |
+====================+====================+====================+==========+
| Torch 2 Compile    | SavedModel         | SavedModel         | TensorRT |
| TorchScript Trace  | TensorFlowTensorRT | TensorFlowTensorRT |          |
| TorchScript Script | ONNX               | ONNX               |          |
| TorchTensorRT      | TensorRT           | TensorRT           |          |
| ONNX               |                    |                    |          |
| TensorRT           |                    |                    |          |
+--------------------+--------------------+--------------------+----------+

**Note:** The Model Navigator has the capability to support any Python function as input.
However, in this particular case, its role is limited to profiling the function without generating any serialized models.

The Model Navigator stores all artifacts within the `navigator_workspace`.
Additionally, it provides the option to save a portable and transferable `Navigator Package` that includes only the models with minimal latency and maximal throughput.
This package also includes base formats that can be used to regenerate the `TensorRT` plan on the target hardware.

**Table:** Model formats that can be generated from saved `Navigator Package` and from model sources.

+------------------------+-----------------------------+
|   From model source    |   From Navigator Package    |
+========================+=============================+
| SavedModel             | TorchTensorRT               |
| TensorFlowTensorRT     | TensorFlowTensorRT          |
| TorchScript Trace      | ONNX                        |
| TorchScript Script     | TensorRT                    |
| Torch 2 Compile        |                             |
| TorchTensorRT          |                             |
| ONNX                   |                             |
| TensoRT                |                             |
+------------------------+-----------------------------+

Installation
--------------

The package can be installed using extra index url:


.. code-block:: text

    pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[<extras,>]


or with nvidia-pyindex:

.. code-block:: text

    pip install nvidia-pyindex
    pip install -U triton-model-navigator[<extras,>]


Extras:

- `tensorflow` - Model Navigator with dependencies for TensorFlow2
- `jax` - Model Navigator with dependencies for JAX

For using with PyTorch no extras are needed.

Quick Start
-------------

Using Model Navigator is as simply as calling `optimize` with `model` and `dataloader`:
The `optimize` function will save all the artifacts it generates in the `navigator_workspace`.

**Note:** The `dataloader` is utilized to determine the maximum and minimum shapes of the inputs utilized during model conversions.
The `Model Navigator` employs a single sample from the `dataloader`, which is then repeated to generate synthetic batches for profiling purposes.
Correctness tests are conducted on a subset of the `dataloader` samples, while verification tests are executed on the entire `dataloader`.

.. code-block:: python

    import logging

    import torch
    import model_navigator as nav

    nav.torch.optimize(
        model=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True).eval(),
        dataloader=[torch.randn(1, 3, 256, 256) for _ in range(10)],
    )

The code snippet below demonstrates the usage of the `PyTritonAdapter` to retrieve the `runner` and other necessary information.
The `runner` serves as an abstraction that connects the model checkpoint with its runtime, making the inference process more accessible and straightforward.
Following that, it initiates the `PyTriton`_  server using the provided parameters.

.. code-block:: python

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

Alternatively, Model Navigator can generate `model_repository` that can be served on the `Triton Inference Server`_:


.. code-block:: python

    nav.triton.model_repository.add_model_from_package(
        model_repository_path=pathlib.Path("model_repository"),
        model_name="resnet50",
        package=package,
        strategy=nav.MaxThroughputStrategy(),
    )


Examples
----------

We provide step-by-step examples that demonstrate how to use various features of Model Navigator.
For the sake of readability and accessibility, we use a simple `torch.nn.Linear` model as an example.
These examples illustrate how to optimize, test and deploy the model on
the PyTriton and Triton Inference Server.

https://github.com/triton-inference-server/model_navigator/tree/main/examples.

Links
-------

* Documentation: https://triton-inference-server.github.io/model_navigator
* Source: https://github.com/triton-inference-server/model_navigator
* Issues: https://github.com/triton-inference-server/model_navigator/issues
* Changelog: https://github.com/triton-inference-server/model_navigator/blob/main/CHANGELOG.md
* Known Issues: https://github.com/triton-inference-server/model_navigator/blob/main/docs/known_issues.md
* Contributing: https://github.com/triton-inference-server/model_navigator/blob/main/CONTRIBUTING.md

.. _Triton Model Navigator: https://github.com/triton-inference-server/model_navigator
.. _Triton Inference Server: https://github.com/triton-inference-server/server
.. _TensorRT: https://github.com/NVIDIA/TensorRT
.. _PyTriton: https://github.com/triton-inference-server/pytriton
