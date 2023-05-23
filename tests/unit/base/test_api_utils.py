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
"""Test for API utils"""

from model_navigator import utilities as nav_utils


def test_unpacked_dataloader_has_valid_len():
    """Test that the UnpackedDataloader has a valid length."""
    dataloader = [1, 2, 3]
    unpacked_dataloader = nav_utils.UnpackedDataloader(dataloader, lambda x: x)
    assert len(unpacked_dataloader) == len(dataloader)


def test_unpacked_dataloader_iterates():
    """Test that the UnpackedDataloader iterates."""
    dataloader = [1, 2, 3]
    unpacked_dataloader = nav_utils.UnpackedDataloader(dataloader, lambda x: x)
    for sample in unpacked_dataloader:
        assert sample in dataloader


def test_unpacked_dataloader_applies_fn():
    """Test that the UnpackedDataloader applies the function."""
    dataloader = [1, 2, 3]
    unpacked_dataloader = nav_utils.UnpackedDataloader(dataloader, lambda x: x + 1)
    for org_sample, sample in zip(dataloader, unpacked_dataloader):
        assert sample == org_sample + 1
