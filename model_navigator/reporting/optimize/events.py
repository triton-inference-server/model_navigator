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
"""Events definition and default emitter."""

from enum import Enum

from pyee import EventEmitter


class OptimizeEvent(str, Enum):
    """All navigator events."""

    MODULE_REGISTERED = "module registered"
    MODULE_REGISTRY_CLEARED = "module registered cleared"

    WORKSPACE_INITIALIZED = "WORKSPACE_INITIALIZED"

    MODULE_PICKED_FOR_OPTIMIZATION = "module picked for optimization"

    OPTIMIZATION_STARTED = "optimization started"
    OPTIMIZATION_FINISHED = "optimization finished"

    PIPELINE_STARTED = "pipeline started"
    PIPELINE_FINISHED = "pipeline finished"
    COMMAND_STARTED = "command started"
    COMMAND_FINISHED = "command finished"

    BEST_MODEL_PICKED = "best model picked"
    MODEL_NOT_OPTIMIZED_ERROR = "model not optimized error"

    INPLACE_STARTED = "inplace started"
    INPLACE_FINISHED = "inplace finished"


_DEFAULT_EVENT_EMITTER = EventEmitter()


def default_event_emitter() -> EventEmitter:
    """Return default event emitter (singleton)."""
    return _DEFAULT_EVENT_EMITTER
