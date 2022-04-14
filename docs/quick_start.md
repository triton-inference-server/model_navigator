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

- [Install the Triton Model Navigator in training environment](#install-the-triton-model-navigator-in-training-environment)
- [Export model](#export-model)
- [Install the Triton Model Navigator in deployment environment](#install-the-triton-model-navigator-in-deployment-environment)
- [Optimize model](#optimize-model)

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

## Install the Triton Model Navigator in training environment

To install latest version of Model Navigator use command:

```shell
$ pip install --extra-index-url https://pypi.ngc.nvidia.com git+https://github.com/triton-inference-server/model_navigator.git [pyt] --upgrade
```

## Export model
This step exports model to all available formats and creates `.nav` package with checkpoints and model meta data.

```shell
$ python

> import model_navigator as nav
> device = "cuda" if torch.cuda.is_available() else "cpu"
> dataloader = [torch.full((3, 5), 1.0, device=device) for _ in range(10)]
> model = torch.nn.Linear(5, 7).to(device).eval()
> pkg_desc = nav.torch.export(model=model, dataloader=dataloader, model_name="my_model")
> pkg_desc.save("my_model.nav")

$ copy my_model.nav deployment-environment

```

## Install the Triton Model Navigator in deployment environment

The recommended way of using the Triton Model Navigator is to build a Docker container with all necessary dependencies:

```shell
$ make docker
```

Run the Triton Model Navigator container from source directory as shown below.
```shell
docker run -it --rm \
 --gpus 1 \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -v ${PWD}:${PWD} \
 -w ${PWD} \
 --net host \
 --name model-navigator \
 model-navigator /bin/bash
```

Learn more about installing the Triton Model Navigator using the instructions in the [Installation](installation.md)
section.

## Optimize model
This step uses previously generated `.nav` package and use it for further conversion and applies optimizations for
Triton Inference Server. In results it produces package that can used directly for deployment on Triton Inference Server.
```shell
$ model-navigator run my_model.nav # conversion + analyzer
$ model-navigator deploy my_model.triton.nav --backend trt
```
