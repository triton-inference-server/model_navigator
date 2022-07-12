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
# Quick Start

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Installation](#installation)
- [Export Model from Source](#export-model-from-source)
- [Optimize for Triton Inference Server](#optimize-for-triton-inference-server)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

The following steps below will guide you through  two-step process of using the Triton Model Navigator to export and
analyze a simple PyTorch model. The instructions assume a directory structure like the following:

```
$HOME
  |--- model_navigator
      |--- docs
      |--- examples
      |--- model_navigator
      |--- tests
      .
      .
      .
```

## Installation

See the [installation](installation.md) guide to install Triton Model Navigator in training environment or use for optimization
on Triton Inference Server.

## Export Model from Source
This step exports model to all available formats and creates `.nav` package with checkpoints and model meta data.

```python
import model_navigator as nav
import torch

model = torch.nn.Linear(5, 7).eval()
dataloader = [torch.full((3, 5), 1.0) for _ in range(10)]
device = "cuda" if torch.cuda.is_available() else "cpu"

pkg_desc = nav.torch.export(
    model=model,
    dataloader=dataloader,
    model_name="my_model",
    target_device=device,
)

```
Next user should verify exported format. User can use custom metrics, measure accuracy, listen to the output etc.
```python
import numpy
import torch

sample_count = 100
valid_outputs = 0
for _ in range(sample_count):
    random_sample = torch.full((3, 5), 1.0)

    # Use source model to generate dummy ground truth
    gt = [model(random_sample).detach().cpu().numpy()]

    feed_dict = {"input__0": random_sample.detach().cpu().numpy()}
    onnx_runner = pkg_desc.get_runner(format=nav.Format.ONNX, runtime=nav.RuntimeProvider.CUDA)
    with onnx_runner:
        output = onnx_runner.infer(feed_dict)

    # Compare output and gt
    for a, b in zip(gt, output.values()):
        if numpy.allclose(a, b, atol=0, rtol=0):
            valid_outputs += 1

accuracy = float(valid_outputs) / float(sample_count)

if accuracy > 0.8:
    pkg_desc.set_verified(format=nav.Format.ONNX, runtime=nav.RuntimeProvider.CUDA)
```
After verification user have to save final Navigator package. This package will contain base format and all information
that allows for recreation of all other formats and rerunning all tests.
```python
nav.save(pkg_desc, "my_model.nav")
```

The `.nav` package can then be copied to the deployment environment.

## Optimize for Triton Inference Server

This step uses previously generated `.nav` package and use it for further conversion and applies optimizations for
Triton Inference Server. In results it produces package that can used directly for deployment on Triton Inference Server.

First, run the model configuration and profiling process.
```shell
$ model-navigator optimize my_model.nav
```

This produces a new "my_model.triton.nav" package. This can then be used as an input to the
`model-navigator select` command, which builds a Triton model repository containing the input model in
its best configuration.

```
$ model-navigator select my_model.triton.nav
```

By default, the configuration is selected only for best throughput.
However other optimization objectives can be specified and combined using weighted ranking.
Additional constraints can be passed to the `select` command,
including bounds on latency, throughput and desired model formats:

```
$ model-navigator select my_model.triton.nav \
                        --objective perf_throughput=10 perf_latency_avg=5  # objectives with weights
                        --max-latency-ms 1  \
                        --min-throughput 100  \
                        --max-gpu-usage-mb 8000  \
                        --target-format trt onnx
```
