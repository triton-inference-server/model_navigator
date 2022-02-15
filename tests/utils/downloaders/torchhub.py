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
from pathlib import Path

from model_navigator.core import DEFAULT_CONTAINER_VERSION
from model_navigator.framework import PyTorch
from tests.utils.common import load_config
from tests.utils.downloader import DownloaderConfig, ModelDownloader

LOGGER = logging.getLogger("model_downloader.torchhub")


def _check_torchhub_downloader_dependencies():
    try:
        # pytype: disable=import-error
        import torch  # noqa
        from polygraphy.common import TensorMetadata  # noqa
        from polygraphy.comparator import DataLoader  # noqa
        from torchvision import transforms  # noqa

        # pytype: enable=import-error

        have_torchhub_dependencies = True
    except ImportError:
        have_torchhub_dependencies = False
    return have_torchhub_dependencies


class TorchHubDownloader(ModelDownloader):
    schemes = ["torchhub"]

    def download_model(self, downloader_config: DownloaderConfig, output_path: Path):
        if self._test_config_path is None:
            raise RuntimeError("Init downloader with path to test config")

        have_torchhub_dependencies = _check_torchhub_downloader_dependencies()
        if have_torchhub_dependencies:
            self._run_locally(downloader_config=downloader_config, output_path=output_path)
        else:
            self._run_in_docker(downloader_config=downloader_config, output_path=output_path)

    def _run_locally(self, downloader_config: DownloaderConfig, output_path: Path):
        _download_model(
            config_path=self._test_config_path, downloader_config=downloader_config, output_path=output_path
        )

    def _run_in_docker(self, downloader_config: DownloaderConfig, output_path: Path):
        from model_navigator.utils.docker import DockerImage

        framework = PyTorch

        test_config = load_config(self._test_config_path)
        script_path = Path(__file__)

        workdir = script_path.parent.parent.parent.parent.resolve()
        assert workdir.name == "model_navigator"

        downloader_kwargs = downloader_config.downloader_kwargs
        mount_as_volumes = [
            script_path.parent.resolve(),
            self._test_config_path.parent.resolve(),
            output_path.parent.resolve(),
            workdir,
            *(Path(p) for p in downloader_kwargs.get("mounts", [])),
        ]
        environment = {"PYTHONPATH": workdir.as_posix(), **downloader_kwargs.get("envs", {})}

        cmd = (
            "bash -c '"
            f"python {script_path.as_posix()} -vvvv "
            f"--config-path {self._test_config_path.resolve()} "
            f"--output-path {output_path.resolve()}"
            "'"
        )

        container_version = test_config.get("container_version", DEFAULT_CONTAINER_VERSION)
        docker_image = DockerImage(f"{framework.image}:{container_version}-{framework.tag}")
        docker_container = docker_image.run_container(
            workdir_path=workdir, environment=environment, mount_as_volumes=mount_as_volumes
        )
        docker_container.run_cmd(cmd)


def _download_model(config_path: Path, downloader_config: DownloaderConfig, output_path: Path):
    import torch  # pytype: disable=import-error

    config_path = Path(config_path)
    test_config = load_config(config_path)

    url = downloader_config.model_url
    kwargs = downloader_config.downloader_kwargs

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
    kwargs.pop("url")

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
        model = _trace(
            device=device,
            model=model,
            normalize_sample_input=normalize_sample_input,
            test_config=test_config,
            precision=precision,
        )
    elif method == "script":
        model = _script(model)
    else:
        raise ValueError(f"Unknown method {method}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(model, output_path.as_posix())
    LOGGER.debug(f"Saving exported TorchScript model to {output_path}")


def _script(model):
    import torch  # pytype: disable=import-error

    model = torch.jit.script(model)
    return model


def _trace(*, device, model, normalize_sample_input, test_config, precision):
    import torch  # pytype: disable=import-error

    def _obtain_shapes(inputs, value_ranges):
        shapes = None
        shapes_to_obtain = ["opt_shapes", "max_shapes", "min_shapes"]
        while not shapes:
            shape_to_obtain = shapes_to_obtain.pop(0)
            shapes = test_config.get(shape_to_obtain, {})
        if not all([inputs, shapes, value_ranges]):
            raise ValueError(
                "Provide inputs, [opt|max|min]_shapes, value_ranges config to enable tracing of PyTorch model"
            )
        return shapes

    def _obtain_dummy_input(inputs, shapes, value_ranges, device, normalize_params):
        import torch  # pytype: disable=import-error
        from polygraphy.common import TensorMetadata
        from polygraphy.comparator import DataLoader
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

    def _update_dtype(spec):
        if precision == "fp16" and "dtype" in spec and spec["dtype"] == "float32":
            spec["dtype"] = "float16"
        return spec

    inputs = test_config.get("inputs", {})
    inputs = {name: _update_dtype(spec) for name, spec in inputs.items()}
    value_ranges = test_config.get("value_ranges", {})
    shapes = _obtain_shapes(inputs, value_ranges)
    dummy_input = _obtain_dummy_input(inputs, shapes, value_ranges, device, normalize_sample_input)
    model = torch.jit.trace_module(model, {"forward": dummy_input})
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--config-path", help="Config path containing downloader args")
    parser.add_argument("--output-path", help="Output path of downloaded model")
    parser.add_argument("--verbose", "-v", action="count", help="Provide verbose output", default=0)
    args = parser.parse_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"

    logging.basicConfig(level=log_level, format=log_format)

    output_path = Path(args.output_path)
    test_config_path = Path(args.config_path)
    test_config = load_config(test_config_path)
    downloader_config = DownloaderConfig.from_test_config(test_config)

    downloader = TorchHubDownloader(test_config_path)
    downloader.download_model(downloader_config=downloader_config, output_path=output_path)


if __name__ == "__main__":
    main()
