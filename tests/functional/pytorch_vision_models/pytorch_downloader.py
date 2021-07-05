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
import logging
from pathlib import Path
from urllib.parse import ParseResult

from polygraphy.common import TensorMetadata
from polygraphy.comparator import DataLoader

from tests.functional.downloader import ModelDownloader

LOGGER = logging.getLogger("model_downloader.torchhub")


class TorchHubDownloader(ModelDownloader):
    scheme = "torchhub"

    def download_model(self, url: ParseResult, output_path: Path, **kwargs):
        import torch  # pytype: disable=import-error

        internal_path = url.path[1:] if url.path.startswith("/") else url.path
        *repo_items, model_name = internal_path.split("/")
        repo = "/".join(repo_items)
        LOGGER.info(f"Loading TorchHub {model_name} model from {repo}")
        LOGGER.info("Downloader kwargs:")
        for key, value in kwargs.items():
            LOGGER.info(f"\t{key} = {value}")

        method = kwargs.pop("method", "trace")
        precision = kwargs.pop("precision", "fp32")
        normalize_sample_input = kwargs.pop("normalize_sample_input", {})

        model = torch.hub.load(repo, model_name, pretrained=True, **kwargs)
        model.eval()
        assert precision.lower() in ["fp32", "fp16"]
        if precision.lower() == "fp16":
            model = model.half()
        if torch.cuda.is_available():
            model.to("cuda")
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"

        LOGGER.debug(f"Model is on {device} device")
        LOGGER.debug(f"Exporting TorchScript model with {method} method")

        if method == "trace":

            def _update_dtype(spec):
                if precision == "fp16" and "dtype" in spec and spec["dtype"] == "float32":
                    spec["dtype"] = "float16"
                return spec

            inputs = self._global_config.get("inputs", {})
            inputs = {name: _update_dtype(spec) for name, spec in inputs.items()}
            value_ranges = self._global_config.get("value_ranges", {})
            shapes = self._obtain_shapes(inputs, value_ranges)
            dummy_input = self._obtain_dummy_input(inputs, shapes, value_ranges, device, normalize_sample_input)
            model = torch.jit.trace_module(model, {"forward": dummy_input})
        elif method == "script":
            model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method {method}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(model, output_path.as_posix())
        LOGGER.debug(f"Saving exported TorchScript model to {output_path}")

    def _obtain_shapes(self, inputs, value_ranges):
        shapes = None
        shapes_to_obtain = ["opt_shapes", "max_shapes", "min_shapes"]
        while not shapes:
            shape_to_obtain = shapes_to_obtain.pop(0)
            shapes = self._global_config.get(shape_to_obtain, {})
        if not all([inputs, shapes, value_ranges]):
            raise ValueError(
                "Provide inputs, [opt|max|min]_shapes, value_ranges config to enable tracing of PyTorch model"
            )
        return shapes

    def _obtain_dummy_input(self, inputs, shapes, value_ranges, device, normalize_params):
        import torch  # pytype: disable=import-error
        from torchvision import transforms  # pytype: disable=import-error

        input_metadata = TensorMetadata()
        for name, spec in inputs.items():
            shape = shapes[name]
            input_metadata.add(name, dtype=spec["dtype"], shape=shape)
        synthetic_dataloader = DataLoader(input_metadata=input_metadata, val_range=value_ranges)
        dummy_sample = synthetic_dataloader[0]

        transforms_list = []
        if normalize_params:
            transforms_list.append(transforms.Normalize(mean=normalize_params["mean"], std=normalize_params["std"]))

        preprocess = transforms.Compose(transforms_list)

        dummy_input = tuple(preprocess(torch.from_numpy(dummy_sample[input_name])).to(device) for input_name in inputs)
        return dummy_input


downloader_cls = TorchHubDownloader
