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

# PyTriton

The [PyTriton](https://github.com/triton-inference-server/pytriton) is a Flask/FastAPI-like interface that simplifies
Triton's deployment in Python environments. In general using PyTriton can serve any Python function. The Model Navigator
provide a `runner` - an abstraction that connects the model checkpoint with its runtime, making the inference process
more accessible and straightforward. The `runner` is a Python API through which an optimized model can serve inference.

## Obtaining runner from Package

The [Navigator Package](../package/package.md) provides an API for obtaining the model for serving inference. One of the
option is to obtain the `runner`:

```python
runner = package.get_runner()
```

The default behavior is to select the model and runner which during profiling obtained the smallest latency and the
largest throughput. This runner is considered as most optimal for serving inference queries. Learn more
about `get_runner`
method in [Navigator Package API](../package/package_api.md).

In order to use the runner in PyTriton additional information for the serving model is required. For that purpose we
provide
a `PyTritonAdapter` that contains all minimal information required to prepare successful deployment of model using
PyTriton.

## Using PyTritonAdapter

Model Navigator provide a dedicated `PyTritonAdapter` to retrieve the `runner` and other information required
to bind a model for serving inference. Following that, you can initialize the PyTriton server using the adapter
information:

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

Once the python script is executed, the model inference is served through HTTP/gRPC endpoints.
