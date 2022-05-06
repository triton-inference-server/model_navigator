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
import numpy
import torch

import model_navigator as nav

device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = [torch.full((3, 5), 1.0, device=device) for _ in range(10)]

model = torch.nn.Linear(5, 7).to(device).eval()

pkg_desc = nav.torch.export(
    model=model,
    dataloader=dataloader,
    override_workdir=True,
)

# Verify ONNX format against model in framework
sample_count = 100
valid_outputs = 0
for _ in range(sample_count):
    random_sample = torch.full((3, 5), 1.0, device="cuda")

    # Use source model to generate dummy ground truth
    gt = [model(random_sample).detach().cpu().numpy()]

    feed_dict = {"input__0": random_sample.detach().cpu().numpy()}
    onnx_runner = pkg_desc.get_runner(format=nav.Format.ONNX, runtime=nav.RuntimeProvider.CUDA)
    with onnx_runner:
        output = onnx_runner.infer(feed_dict)

    # Compare output and gt
    for a, b in zip(gt, output.values()):
        if numpy.allclose(a, b, atol=0, rtol=0):
            valid_outputs += 1

accuracy = float(valid_outputs) / float(sample_count)
print(f"Accuracy: {accuracy}")

if accuracy > 0.8:
    pkg_desc.set_verified(format=nav.Format.ONNX, runtime=nav.RuntimeProvider.CUDA)

# Save nav package
pkg_desc.save("my_model.nav")
