# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import torch

import model_navigator.framework_api as nav

device = "cuda" if torch.cuda.is_available() else "cpu"


def dataloader():
    for _ in range(10):
        yield torch.full((3, 5), 1.0, device=device)


model = torch.nn.Linear(5, 7).to(device).eval()


nav.torch.export(
    model=model,
    dataloader=dataloader,
    override_workdir=True,
)
