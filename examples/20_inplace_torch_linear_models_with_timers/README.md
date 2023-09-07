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

# Optimize pipeline of three Torch linear modles

In this example we show how to build simple pipeline consisting of three Torch linear models and then optimize it with Navigator.

We recommend running this example in NVIDIA NGC [PyTorch containter](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

The Python script `optimize.py` wraps the Python model using Inplace Optimize and then runs it without any chagnes.

To run the original Python code without any modifications run:
```bash
./pass.sh
```

To record the models inputs and run optimizations when enough data has been collected run:
```bash
./optimize.sh
```

To load optimized models and use them in place of the original ones run:
```bash
./run.sh
```