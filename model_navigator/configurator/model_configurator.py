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
import itertools
from collections import Generator

from model_navigator.core import Accelerator, Format, Parameter, Precision

from .variant import Variant


class ModelConfigurator:
    formats = tuple()
    accelerators = (Accelerator.NONE, Accelerator.TRT)
    capture_cuda_graph = (0,)

    def variants(self, model_name: str, max_batch_size: int) -> Generator:
        parameters = [self.formats, self.accelerators, self.capture_cuda_graph]

        combinations = itertools.product(*parameters)

        variants = list()
        for combination in combinations:
            format = combination[0]
            accelerator = combination[1]
            capture_cuda_graph = combination[2]
            if accelerator == Accelerator.TRT:
                precisions = (Precision.FP16, Precision.FP32)
            else:
                precisions = (Precision.ANY,)

            for precision in precisions:
                variant = self._map_on_variant(
                    model_name=model_name,
                    max_batch_size=max_batch_size,
                    format=format,
                    accelerator=accelerator,
                    capture_cuda_graph=capture_cuda_graph,
                    precision=precision,
                )

                variants.append(variant)

        for variant in variants:
            yield variant

    def _map_on_variant(
        self,
        model_name: str,
        format: Format,
        precision: Precision,
        accelerator: Accelerator,
        capture_cuda_graph: int,
        max_batch_size: int,
    ):
        args = [format, accelerator, capture_cuda_graph]
        if precision != Precision.ANY:
            args.append(precision)

        name = self._get_variant_name(model_name, *args)
        variant = Variant(
            name=name,
            format=format,
            precision=precision,
            accelerator=accelerator,
            capture_cuda_graph=capture_cuda_graph,
            gpu_engine_count=1,
            max_batch_size=max_batch_size,
        )

        return variant

    def _get_variant_name(self, model_name, *args):
        params = []
        for item in args:
            item = item if not isinstance(item, Parameter) else item.value
            item = str(item)
            params.append(item)

        suffix = "_".join(params)
        return f"{model_name}.{suffix}"


class TFConfigurator(ModelConfigurator):
    formats = (Format.TF_SAVEDMODEL,)
    accelerators = (
        Accelerator.NONE,
        Accelerator.AMP,
        Accelerator.TRT,
    )


class PyTorchConfigurator(ModelConfigurator):
    formats = (Format.TS_SCRIPT,)
    accelerators = (Accelerator.NONE,)


class ONNXConfigurator(ModelConfigurator):
    formats = (Format.ONNX,)


class TRTConfigurator(ModelConfigurator):
    formats = (Format.TRT,)
    capture_cuda_graph = (0, 1)
    accelerators = (Accelerator.NONE,)
