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
"""ConvertONNX2TRT command."""

from typing import Callable, Iterable, Optional

from model_navigator.api.config import TensorRTProfile
from model_navigator.commands.base import Command
from model_navigator.core.constants import DEFAULT_MAX_BATCH_SIZE_HALVING
from model_navigator.logger import LOGGER


class Convert2TensorRTWithMaxBatchSizeSearch(Command):
    """Command that converts models with conversion max batch size search."""

    @classmethod
    def _execute_conversion(
        cls,
        convert_func: Callable,
        get_args: Callable,
        custom_trt_profile_available: bool,
        batch_dim: Optional[int] = None,
        dataloader_max_batch_size: Optional[int] = None,
        device_max_batch_size: Optional[int] = None,
    ):
        run_search = cls._run_search(
            custom_trt_profile_available=custom_trt_profile_available,
            batch_dim=batch_dim,
            dataloader_batch_size=dataloader_max_batch_size,
            device_max_batch_size=device_max_batch_size,
        )
        if run_search:
            max_conversion_batch_size = (
                cls._execute_conversion_with_max_batch_size_search(  # TODO what is the best value?
                    convert_func=convert_func,
                    get_args=get_args,
                    device_max_batch_size=device_max_batch_size,
                    dataloader_max_batch_size=dataloader_max_batch_size,
                )
            )
            LOGGER.info(f"Converted with maximal batch size: {max_conversion_batch_size}.")
        else:
            LOGGER.info("Search for maximal batch size disable. Execute single conversion.")
            max_conversion_batch_size = dataloader_max_batch_size
            convert_func(get_args())

        return max_conversion_batch_size

    @classmethod
    def _execute_conversion_with_max_batch_size_search(
        cls,
        convert_func: Callable,
        get_args: Callable,
        device_max_batch_size: int,
        dataloader_max_batch_size: int,
    ):
        LOGGER.info("Search for maximal batch size enabled. Execute conversion with adaptive batch size adjustment.")
        max_conversion_batch_size = None
        try:
            for max_batch_size in cls._get_conversion_max_batch_sizes(
                device_max_batch_size,
                dataloader_max_batch_size,
            ):
                convert_func(get_args(max_batch_size=max_batch_size))
                max_conversion_batch_size = max_batch_size
                LOGGER.info(f"Successful conversion for max batch size: {max_conversion_batch_size}.")
        except Exception as e:
            LOGGER.debug(f"Conversion failed with error: {str(e)}")
            if max_conversion_batch_size:
                LOGGER.info(f"Last successful conversion for max batch size: {max_conversion_batch_size}.")
                return max_conversion_batch_size

            fallback_batch_sizes = cls._get_conversion_fallback_batch_sizes(
                device_max_batch_size,
                dataloader_max_batch_size,
            )
            while max_conversion_batch_size is None:
                max_batch_size = next(fallback_batch_sizes)
                try:
                    convert_func(get_args(max_batch_size=max_batch_size))
                    max_conversion_batch_size = max_batch_size
                    LOGGER.info(f"Successfully converted with max batch size: {max_conversion_batch_size}.")
                except Exception as e:
                    LOGGER.debug(f"Conversion failed with error: {str(e)}")
                    if max_batch_size == dataloader_max_batch_size:
                        raise e

        return max_conversion_batch_size

    @staticmethod
    def _get_conversion_max_batch_sizes(
        device_max_batch_size: Optional[int], dataloader_max_batch_size: int
    ) -> Iterable[int]:
        if device_max_batch_size:
            # Temporarily disable max batch size search
            # max_batch_size = device_max_batch_size
            # while max_batch_size < DEFAULT_TENSORRT_MAX_DIMENSION_SIZE:
            #     yield max_batch_size
            #     max_batch_size *= 2
            yield device_max_batch_size
        else:
            yield dataloader_max_batch_size

    @staticmethod
    def _get_conversion_fallback_batch_sizes(
        device_max_batch_size: int, dataloader_max_batch_size: int
    ) -> Iterable[int]:
        max_batch_size_halving_left = DEFAULT_MAX_BATCH_SIZE_HALVING  # TODO what is the best value?
        max_batch_size = device_max_batch_size // 2
        while max_batch_size > dataloader_max_batch_size and max_batch_size_halving_left > 0:
            yield max_batch_size
            max_batch_size //= 2
            max_batch_size_halving_left -= 1
        yield dataloader_max_batch_size

    @classmethod
    def _is_valid_batch_size(cls, batch_size: Optional[int]) -> bool:
        if not batch_size or batch_size < 1:
            return False

        return True

    @classmethod
    def _run_search(
        cls,
        custom_trt_profile_available: bool,
        batch_dim: Optional[int],
        dataloader_batch_size: Optional[int],
        device_max_batch_size: Optional[int],
    ) -> bool:
        if custom_trt_profile_available:
            LOGGER.info("`trt_profile` has been provided by the user.")
            return False

        if batch_dim is None:
            LOGGER.info("`batch_dim` is None. Model does not support batching.")
            return False

        if not cls._is_valid_batch_size(dataloader_batch_size) and not cls._is_valid_batch_size(device_max_batch_size):
            LOGGER.info(
                "Dataloader or device max batch size is invalid.\n"
                "Provided values:\n"
                f"    dataloader_batch_size: {dataloader_batch_size}\n"
                f"    device_max_batch_size: {device_max_batch_size}\n"
            )
            return False

        return True

    @classmethod
    def _get_trt_profile(
        cls,
        dataloader_trt_profile: TensorRTProfile,
        custom_trt_profile: Optional[TensorRTProfile],
    ):
        if not custom_trt_profile:
            LOGGER.info("Using dataloader profile for TRT conversion")
            return dataloader_trt_profile
        else:
            LOGGER.info("Using user specified profile for TRT conversion")
            return custom_trt_profile
