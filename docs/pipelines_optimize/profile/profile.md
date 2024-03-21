<!--
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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

# Profile model or callable

The Triton Model Navigator enhances models and pipelines and provides a uniform method for profiling any Python
function, callable, or model. At present, our support is limited strictly to static batch profiling scenarios.
Profiling is conducted for each sample provided in the dataloader. This allows for the use of various batch sizes in the dataloader, enabling testing of different batch size distributions.
As an example, we will use a simple function that simply sleeps for 50ms:

```python
import time


def custom_fn(input_):
    # wait 50ms
    time.sleep(0.05)
    return input_
```

Let's provide a dataloader we will use for profiling:

```python
# Tuple of batch size and data sample
dataloader = [(1, ["This is example input"])]
```

Finally, run the profiling of the function with prepared dataloader:

```python
nav.profile(custom_fn, dataloader)
```

Review all possible options in the [profile API](api/profile.md).
