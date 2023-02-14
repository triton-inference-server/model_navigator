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

# NVIDIA Deep Learning Examples EfficientNet Model

In this example we show how to optimize EfficientNet model from [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples) repository. We recommend running this example in NVIDIA NGC [TensorFlow2 containter](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow).

## Install dependencies and download model repository

```bash
. ./install.sh
```

## Run optimization


```bash
./optimize.py --model-name EfficientNet-v1-B0
```
