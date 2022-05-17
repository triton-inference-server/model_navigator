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

# Use ONNX model as source model for export
dataloader = [numpy.random.rand(3, 5).astype("float32") for _ in range(10)]
onnx_path = nav_workdir / "onnx_src.nav.workspace" / "onnx" / "model.onnx"
pkg_desc = nav.onnx.export(
    model=onnx_path,
    workdir=nav_workdir,
    dataloader=dataloader,
    override_workdir=True,
    opset=13,
)

# Verify TRT format against ONNX model - ONNX is used as source model.
sample_count = 100
valid_outputs = 0
for _ in range(sample_count):
    random_sample = torch.full((3, 5), 1.0, device="cuda")
    feed_dict = {"input__0": random_sample.detach().cpu().numpy()}

    # Use source ONNX to generate dummy ground truth
    onnx_runner = pkg_desc.get_runner(format=nav.Format.ONNX, runtime=nav.RuntimeProvider.CUDA)
    with onnx_runner:
        gt = onnx_runner.infer(feed_dict)

    trt_runner = pkg_desc.get_runner(
        format=nav.Format.TENSORRT, runtime=nav.RuntimeProvider.TRT, precision=nav.TensorRTPrecision.FP32
    )
    with trt_runner:
        output = trt_runner.infer(feed_dict)

    # Compare output and gt
    for a, b in zip(gt.values(), output.values()):
        # Compare with tolerance - there are numerical differences between source and exported model
        if numpy.allclose(a, b, atol=0.01, rtol=0.01):
            valid_outputs += 1

accuracy = float(valid_outputs) / float(sample_count)
print(f"Accuracy: {accuracy}")

if accuracy > 0.8:
    pkg_desc.set_verified(format=nav.Format.ONNX, runtime=nav.RuntimeProvider.CUDA)

# Save nav package
nav.save(pkg_desc, "my_model.nav")
