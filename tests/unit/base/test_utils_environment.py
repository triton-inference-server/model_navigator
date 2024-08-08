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
from model_navigator.configuration.constants import NAVIGATOR_CONSOLE_OUTPUT_ENV
from model_navigator.utils.environment import get_console_output, use_multiprocessing


def test_get_console_output_return_default_value():
    value = get_console_output()

    assert value == "SIMPLE"


def test_get_console_output_return_upper_cased_value(monkeypatch):
    get_console_output.cache_clear()
    with monkeypatch.context() as m:
        m.setenv(name=NAVIGATOR_CONSOLE_OUTPUT_ENV, value="lowercased")
        value = get_console_output()

        assert value == "LOWERCASED"


def test_use_multiprocessing_return_default_value():
    value = use_multiprocessing()

    assert value is True
