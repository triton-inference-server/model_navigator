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

r"""
Download file from given url to defined path

```shell script
python ./cli/download_file.py \
    --file-url http://models/model.onnx \
    --to-path /mnt/models
```
"""

import argparse
import logging

from model_navigator.args import str2bool
from model_navigator.downloader import Downloader
from model_navigator.log import set_logger, log_dict

LOGGER = logging.getLogger("download_file")


def main():
    parser = argparse.ArgumentParser(description="Download files from given url and store it to defined path.")
    parser.add_argument("--file-url", type=str, required=True, help="Url to file.")
    parser.add_argument("--to-path", type=str, required=True, help="Path where files would be stored.")
    parser.add_argument("-v", "--verbose", help="Provide verbose logs", type=str2bool, default=False)

    args = parser.parse_args()

    set_logger(verbose=args.verbose)
    log_dict("args", vars(args))

    downloader = Downloader(file_url=args.file_url, dst_path=args.to_path)
    downloader.get_file()


if __name__ == "__main__":
    main()
