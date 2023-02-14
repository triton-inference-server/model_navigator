# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
from typing import Any, List, Optional


def dle_convnets_pyt(model_name: str, dataloader: List, max_batch_size: Optional[int] = None, **kwargs: Any):
    from image_classification import models as convnet_models  # pytype: disable=import-error

    import model_navigator as nav

    logger = logging.getLogger(__package__)

    models = {
        "resnet50": convnet_models.resnet50,
        "efficientnet-widese-b0": convnet_models.efficientnet_widese_b0,
    }

    model_cls = models[model_name]

    batch_sizes = None
    if max_batch_size:
        pow2 = math.ceil(math.log(max_batch_size, 2)) + 1
        batch_sizes = [2**n for n in range(pow2)]

    logger.info(f"Testing {model_name}...")
    model = model_cls(pretrained=True).eval()
    package = nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        verbose=True,
        profiler_config=nav.ProfilerConfig(batch_sizes=batch_sizes),
        **kwargs,
    )
    return package
