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
import pytest
from pyee import EventEmitter

from model_navigator.reporting.events import NavigatorEvent


@pytest.fixture
def mock_event_emitter():
    """Mocks default event emitter and records events."""
    emitter = EventEmitter()
    emitter.history = []

    def create_handler(event):  # noqa: D103
        """Function to get event from the closure."""

        def func(*args, **kwargs):
            # store events
            emitter.history.append((event, args, kwargs))

        return func

    for event in iter(NavigatorEvent):
        emitter.on(event, create_handler(event))

    yield emitter
