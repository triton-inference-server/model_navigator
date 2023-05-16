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

# Load and optimize package

In this example, we show how to optimize an existing `.nav` package.
After the optimization is completed, a new package is saved.

We recommend running this example in NVIDIA NGC [PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) or [TensorFlow2](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) container.

For PyTorch nav package run

```bash
./optimize.py --input-path torch_linear.nav --output-path optimized_torch_linear.nav --no-defaults
```

or for TensorFlow2 package:

```bash
./optimize.py --input-path tensorflow_linear.nav --output-path optimized_tensorflow_linear.nav --no-defaults
```

The `--no-defaults` sets the optimize parameters to the package parameters. Default behaviour is to use default parameters.