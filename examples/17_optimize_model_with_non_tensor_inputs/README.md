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

# Optimize model with non-tensor inputs.

In this example we show how to optimize a model with non-tensor inputs. We differentiate two categories of non-tensor inputs:

- tuples, lists and dictionaries - those are fully supported by the Model Navigator, but might not be supported by the underlying technologies (e.g. torchscript).
- primitive types (integers, floats, strings and booleans) - those are currently only supported when they are consistent across all samples in the dataloader. For example if the first sample is `{"input_1": <some tensor>, "input_2": False}` all other samples also must have `input_2` set to `False`. This restriction will be lifted in the future.

In the `optimize.py` both categories are used.

We recommend running this example in NVIDIA NGC [PyTorch containter](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). To run the example simply run the `optimize.py` script:

```bash
./optimize.py
```