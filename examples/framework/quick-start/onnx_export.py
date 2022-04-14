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
from pathlib import Path

import numpy
import torch

import model_navigator as nav

nav_workdir = Path.cwd() / "navigator_workdir"

# Generate source ONNX model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dataloader = [torch.full((3, 5), 1.0, device=device) for _ in range(10)]
model = torch.nn.Linear(5, 7).to(device).eval()
nav.torch.export(
    model=model,
    model_name="onnx_src",
    workdir=nav_workdir,
    dataloader=torch_dataloader,
    target_formats=(nav.Format.ONNX,),
    override_workdir=True,
)

dataloader = [numpy.random.rand(3, 5).astype("float32") for _ in range(10)]
onnx_path = nav_workdir / "onnx_src.nav.workspace" / "onnx" / "model.onnx"
nav.onnx.export(
    model=onnx_path,
    workdir=nav_workdir,
    dataloader=dataloader,
    override_workdir=True,
    opset=13,
)
