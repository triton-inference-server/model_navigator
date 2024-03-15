# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
"""Example client for linear model"""

import logging

import torch  # pytype: disable=import-error
from pytriton.client import ModelClient  # pytype: disable=import-error

logger = logging.getLogger("client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def main():
    """Prepare data and send inference request to PyTriton server"""

    batch_size = 10
    data_size = (5,)
    data_batch = torch.randn(size=(batch_size,) + data_size, dtype=torch.float32).numpy()

    """Use PyTriton client to send inference request."""
    with ModelClient("localhost", "linear") as client:
        logger.info("Sending request")
        result_dict = client.infer_batch(input__0=data_batch)

    logger.info(f"results: {result_dict}")


if __name__ == "__main__":
    main()
