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

import importlib
from typing import Optional

import model_navigator.frameworks.tensorrt.timing_tactics as trt_ttc


def test_timing_manager_should_have_disk_cache_class_registered():
    assert trt_ttc.TimingCacheManager._cache_classes.get(trt_ttc.TimingCacheType.DISK.value) is not None


def test_timing_manager_can_register_new_cache_class():
    @trt_ttc._register_cache_class("test")
    class TestCache(trt_ttc.TimingCache):
        def get(self) -> Optional[trt_ttc.Path]:
            return None

        def save(self):
            pass

    assert trt_ttc.TimingCacheManager._cache_classes.get("test") is not None

    trt_ttc._unregister_cache_class("test")
    assert trt_ttc.TimingCacheManager._cache_classes.get("test") is None


def test_timing_manager_should_provide_cache_name_defaults(mocker, tmp_path):
    mocker.patch(
        "model_navigator.frameworks.tensorrt.timing_tactics.get_gpu_info",
        return_value={"name": "RTX_3090", "cuda_version": "12.100"},
    )
    mocker.patch("model_navigator.frameworks.tensorrt.timing_tactics.get_trt_version", return_value="10.0.0")

    with trt_ttc.TimingCacheManager(model_name="navtest", cache_path=tmp_path / "cache") as cache_file:
        assert cache_file is not None
        assert "rtx_3090" in cache_file.name
        assert "12_100" in cache_file.name
        assert "10_0_0" in cache_file.name
        assert "global" in cache_file.name

        assert not cache_file.exists()


def test_timing_manager_should_provide_cache_name_per_model(mocker, tmp_path):
    mocker.patch("model_navigator.frameworks.tensorrt.timing_tactics.get_trt_version", return_value="10.0.0")
    with trt_ttc.TimingCacheManager(
        model_name="navtest", cache_path=tmp_path / "cache", strategy=trt_ttc.TimingCacheStrategy.PER_MODEL
    ) as cache_file:
        assert cache_file.name.startswith("navtest_")


def test_timing_manager_should_provide_cache_name_per_model_env(mocker, tmp_path):
    mocker.patch(
        "os.environ", {"MODEL_NAVIGATOR_TENSORRT_TIMING_CACHE_STRATEGY": trt_ttc.TimingCacheStrategy.PER_MODEL.value}
    )
    importlib.reload(trt_ttc)

    mocker.patch("model_navigator.frameworks.tensorrt.timing_tactics.get_trt_version", return_value="10.0.0")
    with trt_ttc.TimingCacheManager(model_name="navtest", cache_path=tmp_path / "cache") as cache_file:
        assert cache_file.name.startswith("navtest_")


def test_timing_manager_should_provide_cache_name_user(mocker, tmp_path):
    # FIXME(kn): it is unclear how to detect user is providing a directory or a file
    # FIXME(kn): it automatically switches to USER mode, thats not good
    # FIXME(kn): should user always provide a file?
    mocker.patch("model_navigator.frameworks.tensorrt.timing_tactics.get_trt_version", return_value="10.0.0")
    with trt_ttc.TimingCacheManager(model_name="navtest", cache_path=tmp_path / "my.cache") as cache_file:
        assert cache_file.name == "my.cache"


def test_timing_manager_should_provide_none():
    with trt_ttc.TimingCacheManager(strategy=trt_ttc.TimingCacheStrategy.NONE) as cache_file:
        assert cache_file is None


def test_timing_manager_should_provide_same_cache(mocker, tmp_path):
    mocker.patch("model_navigator.frameworks.tensorrt.timing_tactics.get_trt_version", return_value="10.0.0")
    opts = {"cache_path": tmp_path / "cache", "strategy": trt_ttc.TimingCacheStrategy.PER_MODEL}

    with trt_ttc.TimingCacheManager(model_name="navtest1", **opts) as cache_file:
        cache_file.write_text("navtest1")

    with trt_ttc.TimingCacheManager(model_name="navtest2", **opts) as cache_file:
        cache_file.write_text("navtest2")

    with trt_ttc.TimingCacheManager(model_name="navtest1", **opts) as cache_file:
        assert cache_file.read_text() == "navtest1"
