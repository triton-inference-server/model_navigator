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

import click

from model_navigator.downloader import Downloader
from model_navigator.log import log_dict, set_logger

LOGGER = logging.getLogger("download_file")


@click.command(name="download-model", help="Download model from given uri and store it to defined path.")
@click.option("--model-uri", required=True, help="Uri to model file or archive.")
@click.option(
    "--save-to",
    type=click.Path(writable=True),
    required=True,
    help="Path where model file or archive has to be downloaded.",
)
@click.option("-v", "--verbose", help="Provide verbose logs.", default=False, type=bool, is_flag=True)
def download_cmd(model_uri: str, save_to: str, verbose: bool):
    set_logger(verbose=verbose)
    if verbose:
        log_dict("args", {"model_uri": model_uri, "save_to": save_to, "verbose": verbose})

    downloader = Downloader(file_uri=model_uri, dst_path=save_to)
    downloader.get_file()
