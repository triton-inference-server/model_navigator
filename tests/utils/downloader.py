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
import io
import logging
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import ParseResult, urlparse
from urllib.request import urlopen

from model_navigator.framework import PyTorch, TensorFlow2

LOGGER = logging.getLogger(__name__)

_SUFFIX2FRAMEWORK = {
    ".savedmodel": TensorFlow2,
    ".plan": PyTorch,
    ".onnx": PyTorch,
    ".pt": PyTorch,
}

DOWNLOADERS = {}


@dataclass
class DownloaderConfig:
    model_url: ParseResult
    downloader_kwargs: Dict[str, Any]

    @classmethod
    def from_test_config(cls, config: Dict[str, Any]):
        model_downloader_kwargs = config["model_downloader_kwargs"]
        model_url = urlparse(model_downloader_kwargs["url"])
        return cls(model_url, model_downloader_kwargs)


class ModelDownloader(ABC):
    def __init__(self, test_config_path: Optional[Path] = None):
        self._test_config_path = test_config_path

    @abstractmethod
    def download_model(self, downloader_config: DownloaderConfig, output_path: Path):
        pass


class TarModelDownloader(ModelDownloader):
    """Download tar-compressed models from url and extract to output_path."""

    schemes = ["http", "https"]

    def download_model(self, downloader_config: DownloaderConfig, output_path: Path):
        url = downloader_config.model_url
        resp = urlopen(url.geturl())
        tar = tarfile.open(fileobj=io.BytesIO(resp.read()))
        tar.extractall(output_path)


def _fill_downloaders_registry():
    from tests.utils.downloaders.torchhub import TorchHubDownloader

    downloaders = [TorchHubDownloader, TarModelDownloader]

    try:
        from tests.utils.downloaders.internal import InternalTestDataDownloader  # pytype: disable=import-error

        downloaders += [InternalTestDataDownloader]
    except ImportError:
        pass

    for DownloaderCls in downloaders:
        for scheme in DownloaderCls.schemes:
            if scheme in DOWNLOADERS:
                raise RuntimeError(
                    f"Multiple downloaders handling {scheme} scheme: {DOWNLOADERS[scheme]} and {DownloaderCls}"
                )
            DOWNLOADERS[scheme] = DownloaderCls

    LOGGER.debug(
        f"Available downloaders: "
        f"{', '.join([f'{scheme}={DownloaderCls}' for scheme, DownloaderCls in DOWNLOADERS.items()])}"
    )


_fill_downloaders_registry()
