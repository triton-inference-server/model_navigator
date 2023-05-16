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

# NVIDIA Deep Learning Examples ResNet50 Model

In this example we show how to optimize ResNet50 model from [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples) repository. We recommend running this example in NVIDIA NGC [PyTorch containter](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

## Install dependencies and download the model repository and the dataset

```bash
. ./install.sh
```

This script downloads the model repository and dummy version of ImageNet dataset. To use the origina ImageNet dataset on which the model was trained on please follow [instructions](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md#2-download-and-preprocess-the-dataset) from the model repository and in the next step specify the dataset directory with `--data-path` argument.

## Run optimization


```bash
./optimize.py
```
