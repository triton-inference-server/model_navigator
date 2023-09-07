#!/usr/bin/env python3
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
from typing import Tuple

import torch  # pytype: disable=import-error

import model_navigator as nav


class ModelWithNonTensorInputs(torch.nn.Module):
    """Model with non-tensor inputs"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 7)

    def forward(self, xy: Tuple[torch.Tensor, torch.Tensor], raise_error: bool):
        """Forward pass"""
        x, y = xy
        if raise_error:
            raise ValueError("Error")
        return self.linear(x) + y


def main():
    """Get model, dataloader, and run optimization"""
    model = ModelWithNonTensorInputs()
    dataloader = [((torch.randn(3, 5), torch.randn(3, 7)), False) for _ in range(10)]

    """
    Optimize the model by performing model export, conversion, correctness tests, and profiling.

    Results are saved in `navigator_workspace` directory.
    """
    nav.torch.optimize(
        model=model,
        dataloader=dataloader,
    )


if __name__ == "__main__":
    main()
