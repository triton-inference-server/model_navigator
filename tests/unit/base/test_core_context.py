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
def test_context_store_values_between_imports():
    def get_empty():
        from model_navigator.core.context import global_context

        return global_context.get("foo")

    def set_value():
        from model_navigator.core.context import global_context

        global_context.set("foo", "bar")
        return global_context.get("foo")

    def get_value():
        from model_navigator.core.context import global_context

        return global_context.get("foo")

    def pop_value():
        from model_navigator.core.context import global_context

        return global_context.pop("foo")

    assert get_empty() is None
    assert set_value() == "bar"
    assert get_value() == "bar"
    assert pop_value() == "bar"
    assert get_empty() is None


def test_context_clear_remove_all_items_stored_in_context():
    from model_navigator.core.context import global_context

    global_context.set("foo", "bar")
    assert global_context.get("foo") == "bar"

    global_context.clear()
    assert global_context.get("foo") is None
