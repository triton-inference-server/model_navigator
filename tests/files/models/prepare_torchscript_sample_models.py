#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# pytype: disable=import-error
import torch
from torch import nn

# pytype: enable=import-error


class Identity(nn.Module):
    def forward(self, x):
        return x


def prepare_scripted_identity(output_path: str):
    model = Identity()
    model = torch.jit.script(model)
    torch.jit.save(model, output_path)


def prepare_traced_identity(output_path: str):
    model = Identity()
    dummy_input = tuple(torch.zeros(128, 3, 240, 240, dtype=torch.float, device="cpu"))
    model = torch.jit.trace_module(model, {"forward": dummy_input})
    torch.jit.save(model, output_path)


def main():
    prepare_scripted_identity(output_path="tests/files/models/identity.scripted.pt")
    prepare_traced_identity(output_path="tests/files/models/identity.traced.pt")


if __name__ == "__main__":
    main()
