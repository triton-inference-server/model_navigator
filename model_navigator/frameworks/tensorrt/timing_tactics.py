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
"""TensorRT Timing Tactics Cache.

Cache management for TensorRT Timing Tactics.

Layer Timing Cache is used for optimizing build performance by storing and reusing information about timing tactics.

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#timing-cache

Example:
```
from model_navigator.frameworks.tensorrt.timing_tactics import TimingCacheManager

with TimingCacheManager() as cache_path:
    engine = engine_from_network(network,
        config=CreateConfig(
            ...,
            load_timing_cache=cache_path.as_posix() if cache_path or None,
        ),
        save_timing_cache=cache_path,
    )

```

You can use your own cache class if you implement the `TimingCache` abstract class and register it with the
`_register_cache_class` decorator.

Example:
```
@_register_cache_class("MyCache")
class MyCache(TimingCache):
    def get(self) -> Optional[Path]:
        return None

    def save(self):
        pass

with TimingCacheManager(cache_type="MyCache") as cache_path:
    pass
```

To change default behavior, you can set the environment variables:
* NAV_TRT_TIMING_CACHE_TYPE - default cache type (disk)
* NAV_TRT_TIMING_CACHE_STRATEGY - default cache strategy (global)

Example:
```
os.environ["NAV_TRT_TIMING_CACHE_TYPE"] = "none" # turn off tactics cache
```

"""

import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from model_navigator.frameworks.tensorrt.utils import get_version as get_trt_version
from model_navigator.inplace.config import DEFAULT_CACHE_DIR
from model_navigator.utils.environment import get_gpu_info


class TimingCacheType(Enum):
    """Implementations types of the timing cache manager."""

    DISK = "disk"
    """Simply provides correct paths base on the parameters"""


class TimingCacheStrategy(Enum):
    """Cache storage strategies for the timing tactics cache manager."""

    GLOBAL = "global"
    """This will speed up all TRT optimizations."""

    PER_MODEL = "per_model"
    """This will speed up subsequent TRT optimizations of the same model."""

    NONE = "none"
    """No cache is provided."""

    USER = "user"
    """When user specifies the cache path."""


DEFAULT_CACHE_TYPE = os.environ.get("MODEL_NAVIGATOR_TENSORRT_TIMING_CACHE_TYPE", TimingCacheType.DISK.value)
#: Default cache type of the timing tactics cache manager.
#: It can be changed using environment variable NAV_TRT_TIMING_CACHE_TYPE.

DEFAULT_CACHE_STRATEGY = TimingCacheStrategy(
    os.environ.get("MODEL_NAVIGATOR_TENSORRT_TIMING_CACHE_STRATEGY", TimingCacheStrategy.GLOBAL.value)
)
#: Default cache strategy of the timing tactics cache manager.
#:  It can be changed using environment variable TRT_TIMING_CACHE_STRATEGY.


class TimingCache(ABC):
    """Abstract class for the timing cache manager."""

    def __init__(self, model_name: str, cache_path: Optional[Path], strategy: TimingCacheStrategy):
        """Initialize the TimingCache class.

        Args:
            model_name (str): Model name, used for per model caching strategy.
            cache_path (Optional[Path], optional): Where the cache is stored.
            strategy (TimingCacheStrategy, optional): See `TimingCacheStrategy`.
        """
        self.model_name = model_name or "global"
        self.cache_path = cache_path or DEFAULT_CACHE_DIR
        self.strategy = strategy

    @abstractmethod
    def get(self) -> Optional[Path]:
        """Extracts and prepares correct cache file and returns path."""

    @abstractmethod
    def save(self) -> None:
        """Stores the cache file after the model optimization if needed."""

    def _get_trt_timing_cache_name(self, prefix: str) -> str:
        """Name of the TensorRT timing cache file.

        Cache is made per GPU, CUDA and TRT version.

        https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#:~:text=The%20information%20it%20contains%20is%20specific%20to%20the%20targeted%20device%2C%20CUDA%2C%20TensorRT%20versions%2C%20and%20BuilderConfig
        Args:
            prefix (str): Prefix of the cache file.

        Returns:
            str: file name
        """
        prefix = f"{self._fixing_name_for_path(prefix).lower()}_"

        gpu_info = get_gpu_info()

        gpu_name = self._fixing_name_for_path(gpu_info.get("name", "n_a")).lower()
        cuda_version = self._fixing_name_for_path(gpu_info.get("cuda_version", "n_a"))
        trt_version = self._fixing_name_for_path(str(get_trt_version()).replace(".", "_"))

        return f"{prefix}{gpu_name}_cuda_{cuda_version}_trt_{trt_version}.cache"

    def _fixing_name_for_path(self, name):
        return name.replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


ITimingCache = TypeVar("ITimingCache", bound=TimingCache)


class DiskTimingCache(TimingCache):
    """Manages the timing cache files created by TRT."""

    def __init__(self, model_name: str, cache_path: Optional[Path], strategy: TimingCacheStrategy):
        """Initialize the DiskTimingCache class.

        If user provide existing file or file that ends with .cache, it will switch to USER strategy.

        Args:
            model_name (str): model name, "tag" for caching file for per model strategy.
            cache_path (Optional[Path]): User provided cache file path (file suffix must be .cache) or directory.
            strategy (TimingCacheStrategy): See `TimingCacheStrategy`.
        """
        super().__init__(model_name, cache_path, strategy)
        self._prepare_cache()

    def _prepare_cache(self):
        if self.cache_path.is_file() or self.cache_path.suffix == ".cache":
            # user provided a file path, switching to USER strategy
            self.strategy = TimingCacheStrategy.USER
            return

        self.cache_path.mkdir(parents=True, exist_ok=True)

    def get(self) -> Optional[Path]:
        """Based on the strategy, returns the correct cache file path."""
        if self.strategy == TimingCacheStrategy.NONE:
            return None

        if self.strategy == TimingCacheStrategy.USER:
            return self.cache_path

        if self.strategy == TimingCacheStrategy.GLOBAL:
            return self.cache_path / self._get_trt_timing_cache_name("global")

        return self.cache_path / self._get_trt_timing_cache_name(self.model_name)

    def save(self):
        """Already saved in the correct path."""
        pass


class TimingCacheManager:
    """Manages the cache of timing tactics for TensorRT.

    There are three types of management strategies:
    * global - one cache for all models and layers
    * per_model - one cache per model
    * user - user specified cache file
    * none - no cache is provided

    This is context manager as it loads and saves the caches from the disk or user provided cache class

    Attributes:
        cache (TimingCache): Am object of timing tactics cache.
    """

    _cache_classes: Dict[str, Any] = {}
    """Registered cache classes."""

    def __init__(
        self,
        model_name: str = "",
        cache_path: Optional[Path] = None,
        strategy: TimingCacheStrategy = DEFAULT_CACHE_STRATEGY,
        cache_type: str = DEFAULT_CACHE_TYPE,
    ):
        """Initialize the TimingCacheManager class.

        Args:
            model_name (str): Model name, used for per model caching strategy.
            cache_path (Optional[Path], optional): Where the cache is stored. Defaults to None.
            strategy (TimingCacheStrategy, optional): See `TimingCacheStrategy`. Defaults to DEFAULT_CACHE_STRATEGY.
            cache_type (str, optional): see `TimingCacheType`. Defaults to DEFAULT_CACHE_TYPE.
        """
        self.cache = self._get_cache_by_type(cache_type)(model_name, cache_path, strategy)

    def _get_cache_by_type(self, cache_type: str) -> Type[ITimingCache]:
        if cache_class := self._cache_classes.get(cache_type):
            return cache_class
        else:
            raise NotImplementedError(
                f"Unsupported cache_class '{cache_type}'. Register a new cache class using `register_cache_class`."
            )

    def __enter__(self) -> Optional[Path]:
        """Prepares the timing cache and returns the cache path."""
        return self.cache.get()

    def __exit__(self, _exc_type, exc_value, _traceback):
        """Exit the context manager."""
        if exc_value is None:
            self.cache.save()
        else:
            raise exc_value


def _register_cache_class(name: str):
    """Class decorator to register a new cache class.

    Class must implement the `TimingCache` abstract class.

    Args:
        name (str): Name of the class to use for identification.
    """

    def _decorate(cache_class: Type[ITimingCache]):
        if issubclass(cache_class, TimingCache):
            TimingCacheManager._cache_classes[name] = cache_class
        else:
            raise ValueError("Cache class must be derived from TimingCache.")

        return cache_class

    return _decorate


def _unregister_cache_class(name: str):
    """Unregister a cache class by name.

    Args:
        name (str): Name of the class to unregister.
    """
    if name in TimingCacheManager._cache_classes:
        del TimingCacheManager._cache_classes[name]


_register_cache_class(TimingCacheType.DISK.value)(DiskTimingCache)
