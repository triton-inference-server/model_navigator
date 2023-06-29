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

# Navigator Package

The model graph and/or checkpoint is not enough to perform a successful deployment of the model. When you are deploying
model for inference you need to be aware of model inputs and outputs definition, maximal batch size that can be used
for inference and other.

On that purpose we have created a `Navigator Package` - an artifact containing the serialized model, model metadata and
optimization details.

The `Navigator Package` is recommended way of sharing the optimized model for deployment
on [PyTriton](../pytriton/pytriton_deployment.md) or [Triton Inference Server](../triton/triton_deployment.md) sections
or re-running the `optimize` method on different hardware.

## Save

The package created during [optimize](../optimize/optimize.md) can be saved in form of Zip file using the API method:

```python
import model_navigator as nav

nav.package.save(
    package=package,
    path="/path/to/package.nav"
)
```

The `save` method collect the generated models from workspace selecting:

- base formats - first available serialization formats exporting model from source
- max throughput format - the model that achieved the highest throughput during profiling
- min latency format - the model that achieved the minimal latency during profiling

Additionally, the package contains:

- status file with optimization details
- logs from optimize execution
- reproduction script per each model format
- input and output data samples in form on numpy files

Read more in [save method API specification](package_load_api.md).

## Load

The packages saved to file can be loaded for further processing:

```python
import model_navigator as nav

package = nav.package.load(
    path="/path/to/package.nav"
)
```

Once the package is loaded you can obtain desired information or use it to `optimize` or `profile` the package. Read
more in [load method API specification](package_load_api.md).

## Optimize

The loaded package object can be used to re-run the optimize process. In comparison to the framework dedicated API, the
package optimize process start from the serialized models inside the package and reproduce the available optimization
paths. This step can be used to reproduce the process without access to sources on different hardware.

The optimization from package can be run using:

```python
import model_navigator as nav

optimized_package = nav.package.optimize(
    package=package
)
```

At the end of the process the new optimized models are generated. Please mind, the workspace is overridden in this step.
Read more in [optimize method API specification](package_optimize_api.md).

## Profile

The optimize process use a single sample from dataloader for profiling. The process is focusing on selecting the best
model format and this requires an unequivocal sample for performance comparison.

In some cases you may want to profile the models on different dataset. On that purpose the Model Navigator
expose the API for profiling all samples in dataset for each model:

```python
import torch
import model_navigator as nav

profiling_results = nav.package.profile(
    package=package,
    dataloader=[torch.randn(1, 3, 256, 256), torch.randn(1, 3, 512, 512)],
)
```

The results contain profiling information per each model and sample. You can use it to perform desired analysis based
on the results. Read more in [profile method API specification](package_profile_api.md).
