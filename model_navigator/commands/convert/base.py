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

from typing import Callable, Generator, Optional

from model_navigator.commands.base import Command
from model_navigator.core.constants import DEFAULT_MAX_BATCH_SIZE_HALVING
from model_navigator.core.logger import LOGGER


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
        """Execution conversion.

        The method verify if the search for maximal batch size is enabled in process and execute appropriate strategy
        for conversion - with or without fallback.

        Args:
            convert_func: A callable that implements conversion to given format
            get_args: A callable that provide arguments required by convert_func
            custom_trt_profile_available: Indicate if user provided custom profiles for TensorRT
            batch_dim: Provide the place where batch shape is stored in shapes
            device_max_batch_size: Maximal batch size found for device
            dataloader_max_batch_size: Batch size used by the dataloader

        Returns:
            New max batch size for which conversion succeeded

        Raises:
            A conversion exception when command and fallback failed.
        """
        run_search = cls._run_search(
            custom_trt_profile_available=custom_trt_profile_available,
            batch_dim=batch_dim,
            dataloader_batch_size=dataloader_max_batch_size,
            device_max_batch_size=device_max_batch_size,
        )

        if run_search:
            assert dataloader_max_batch_size is not None
            assert device_max_batch_size is not None

            conversion_max_batch_size = (
                cls._execute_conversion_with_max_batch_size_search(  # TODO what is the best value?
                    convert_func=convert_func,
                    get_args=get_args,
                    device_max_batch_size=device_max_batch_size,
                    dataloader_max_batch_size=dataloader_max_batch_size,
                )
            )
        else:
            conversion_max_batch_size = cls._execute_single_conversion(
                convert_func=convert_func, get_args=get_args, max_batch_size=dataloader_max_batch_size
            )
        LOGGER.info(f"Converted with maximal batch size: {conversion_max_batch_size}.")

        return conversion_max_batch_size

    @classmethod
    def _execute_single_conversion(
        cls,
        convert_func: Callable,
        get_args: Callable,
        max_batch_size: int,
    ):
        LOGGER.info("Search for maximal batch size disable. Execute single conversion.")
        convert_func(get_args())
        return max_batch_size

    @classmethod
    def _execute_conversion_with_max_batch_size_search(
        cls,
        convert_func: Callable,
        get_args: Callable,
        device_max_batch_size: int,
        dataloader_max_batch_size: int,
    ):
        """Execution of conversion with max batch size search.

        The method try to convert the model with device or dataloader max batch size and provide a fallback logic
        when the conversion fails.

        Args:
            convert_func: A callable that implements conversion to given format
            get_args: A callable that provide arguments required by convert_func
            device_max_batch_size: Maximal batch size found for device
            dataloader_max_batch_size: Batch size used by the dataloader

        Returns:
            New max batch size for which conversion succeeded

        Raises:
            A conversion exception when command and fallback failed.
        """
        LOGGER.info("Search for maximal batch size enabled. Execute conversion with adaptive batch size adjustment.")
        try:
            max_batch_size = cls._get_conversion_max_batch_sizes(device_max_batch_size, dataloader_max_batch_size)
            convert_func(get_args(max_batch_size=max_batch_size))
            conversion_max_batch_size = max_batch_size
            LOGGER.info(f"Successful conversion for max batch size: {conversion_max_batch_size}.")
        except Exception as e:
            LOGGER.debug(f"Conversion failed with error: {str(e)}.Trying fallback with smaller batch sizes.")
            conversion_max_batch_size = cls._fallback_conversion(
                convert_func, get_args, device_max_batch_size, dataloader_max_batch_size
            )

        return conversion_max_batch_size

    @classmethod
    def _fallback_conversion(
        cls,
        convert_func: Callable,
        get_args: Callable,
        device_max_batch_size: int,
        dataloader_max_batch_size: int,
    ):
        """Fallback conversion.

        When converting with selected device_max_batch_size has failed, the fallback is adapting the max batch size,
        update the TensorRT profiles and retry the process.

        The process stops when:
        - there is successful conversion
        - the number of retry has exceeded
        - the next selected batch size is the dataloader batch size

        Args:
            convert_func: A callable that implements conversion to given format
            get_args: A callable that provide arguments required by convert_func
            device_max_batch_size: Maximal batch size found for device
            dataloader_max_batch_size: Batch size used by the dataloader

        Returns:
            New max batch size for which conversion succeeded

        Raises:
            A conversion exception when command failed.
        """
        conversion_max_batch_size = None
        fallback_batch_sizes = cls._get_conversion_fallback_batch_sizes(
            device_max_batch_size,
            dataloader_max_batch_size,
        )
        while conversion_max_batch_size is None:
            max_batch_size = next(fallback_batch_sizes)
            try:
                convert_func(get_args(max_batch_size=max_batch_size))
                conversion_max_batch_size = max_batch_size
                LOGGER.info(f"Successfully converted with max batch size: {conversion_max_batch_size}.")
            except Exception as e:
                LOGGER.debug(f"Conversion failed with error: {str(e)}")
                if max_batch_size == dataloader_max_batch_size:
                    raise e

        return conversion_max_batch_size

    @staticmethod
    def _get_conversion_max_batch_sizes(device_max_batch_size: Optional[int], dataloader_max_batch_size: int) -> int:
        """Select the batch size for first conversion attempt."""
        if device_max_batch_size:
            return device_max_batch_size
        else:
            return dataloader_max_batch_size

    @staticmethod
    def _get_conversion_fallback_batch_sizes(
        device_max_batch_size: int, dataloader_max_batch_size: int
    ) -> Generator[int, None, None]:
        """Calculate the fallback batch size.

        The strategy is to split the current max batch size // 2 until the threshold is exceeded or the value is
        equal to dataloader provided by user.
        """
        max_batch_size_halving_left = DEFAULT_MAX_BATCH_SIZE_HALVING  # TODO what is the best value?
        max_batch_size = device_max_batch_size // 2
        while max_batch_size > dataloader_max_batch_size and max_batch_size_halving_left > 0:
            yield max_batch_size
            max_batch_size //= 2
            max_batch_size_halving_left -= 1
        yield dataloader_max_batch_size

    @classmethod
    def _is_valid_batch_size(cls, batch_size: Optional[int]) -> bool:
        """Validate if provided batch size is acceptable."""
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
        """Validate if conversion if fallback strategy has to be ran."""
        if custom_trt_profile_available:
            LOGGER.info("`trt_profile` has been provided by the user.")
            return False

        if batch_dim is None:
            LOGGER.info("`batch_dim` is None. Model does not support batching.")
            return False

        if not cls._is_valid_batch_size(dataloader_batch_size) or not cls._is_valid_batch_size(device_max_batch_size):
            LOGGER.info(
                "Dataloader or device max batch size is invalid.\n"
                "Provided values:\n"
                f"    dataloader_batch_size: {dataloader_batch_size}\n"
                f"    device_max_batch_size: {device_max_batch_size}\n"
            )
            return False

        return True
