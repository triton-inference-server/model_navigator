# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from io import StringIO

import pytest
from loguru import logger
from packaging.version import Version

from model_navigator.utils import module

trt = module.lazy_import("tensorrt")


def trt_missing_or_old_version():
    """Returns True if trt is not installed or in the old version."""
    if hasattr(trt, "__version__"):
        return Version(trt.__version__) < Version("8.0")
    else:
        return True


@pytest.mark.skipif(trt_missing_or_old_version(), reason="trt is not installed or in the old version")
def test_trt_logger():
    # given
    from model_navigator.frameworks.tensorrt.utils import get_trt_logger

    # setup loguru to log into stream
    log_stream = StringIO()
    logger.remove()
    logger.add(log_stream, format="{level}: {message}", level="DEBUG")

    global TRT_LOGGER
    TRT_LOGGER = None  # force creating new logger
    trt_logger = get_trt_logger()

    # when
    for level in [
        trt.Logger.INTERNAL_ERROR,
        trt.Logger.ERROR,
        trt.Logger.WARNING,
        trt.Logger.INFO,
        trt.Logger.VERBOSE,
    ]:
        trt_logger.log(level, str(level))

    # then
    logs = log_stream.getvalue()
    assert logs == "ERROR: Severity.INTERNAL_ERROR\nERROR: Severity.ERROR\nWARNING: Severity.WARNING\n"
