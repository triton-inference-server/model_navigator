#!/usr/bin/env python3
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
import logging
from typing import Dict, Iterable

import numpy as np

# pytype: disable=import-error
import torch

# pytype: enable=import-error
from model_navigator.converter.dataloader import Dataloader
from model_navigator.model import ModelSignatureConfig
from model_navigator.utils.signature import load_annotation

LOGGER = logging.getLogger("ts2onnx")


def convert(input_model_path, output_path, opset, dataloader: Dataloader, verbose):
    model = torch.jit.load(input_model_path.as_posix())
    model.eval()

    # TODO: on which device (which cuda idx) we should run conversion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    LOGGER.debug(f"Model is on {device} device")

    io_spec: ModelSignatureConfig = load_annotation(input_model_path)

    def _extract_dyn_axes(spec):
        indexes = np.where(np.array(spec.shape) == -1)[0]
        return list(map(int, indexes))

    inputs = io_spec.inputs or {}
    outputs = io_spec.outputs or {}
    dynamic_axes: Dict[str, Iterable[int]] = {
        **{spec.name: _extract_dyn_axes(spec) for name, spec in inputs.items()},
        **{spec.name: _extract_dyn_axes(spec) for name, spec in outputs.items()},
    }

    # FIXME: argument handling here is really awful, unintuitive and can break easily.
    # Ideally, Triton could support kwargs in the model's forward method,
    # which has been recently implemented in libtorch.
    input_names = [name for name, spec in inputs.items()]
    output_names = [name for name, spec in outputs.items()]
    model_args = tuple(torch.from_numpy(i).to(device=device) for _, i in next(iter(dataloader)).items())

    torch.onnx.export(
        model=model,
        args=model_args,
        f=output_path,
        opset_version=opset,
        verbose=verbose,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
