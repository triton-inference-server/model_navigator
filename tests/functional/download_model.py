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
import logging
from pathlib import Path
from urllib.parse import urlparse

import yaml

from tests.functional.utils import load_file

LOGGER = logging.getLogger("download_model")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--downloader-path", help="Path to downloader file")
    parser.add_argument("--config-path", help="Config path containing downloader args")
    parser.add_argument("--output-path", help="Output path of downloaded model")
    parser.add_argument("--verbose", "-v", action="count", help="Provide verbose output", default=0)
    args = parser.parse_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"

    logging.basicConfig(level=log_level, format=log_format)

    downloader_cls = load_file(args.downloader_path, "downloader_cls", "downloader_cls")

    config_path = Path(args.config_path)
    src_model_path = Path(args.output_path)
    with config_path.open("r") as config_file:
        config = yaml.safe_load(config_file)

    model_downloader_kwargs = config.pop("model_downloader_kwargs")
    model_url = urlparse(model_downloader_kwargs.pop("url"))
    downloader = downloader_cls(config)
    downloader.download_model(model_url, src_model_path, **model_downloader_kwargs)


if __name__ == "__main__":
    main()
