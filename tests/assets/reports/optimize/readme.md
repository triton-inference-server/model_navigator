<!--
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

This folder contains recorded events for various pipeline use cases and expected report results.

The events were recorded with the following script:

```python
# pytype: skip-file

import atexit
from datetime import datetime
from multiprocessing import current_process

from model_navigator.reporting.optimize.events import OptimizeEvent, default_event_emitter


def create_handler(event):  # noqa: D103
    global events

    def func(*args, **kwargs):
        events.append((event, args, kwargs))

    return func


def save_workspace(path):  # noqa: D103
    global workspace
    now = datetime.now()
    datetime_str = now.strftime("_%Y%m%d_%H%M%S") + f".{now.microsecond // 1000:03d}"
    workspace = path.parent.name + datetime_str


def store_events():  # noqa: D103
    global events, workspace

    with open(f"events_{workspace}.txt", "w") as fp:
        for event, _, kwargs in events:
            if 'status' in kwargs:
                status = str(kwargs['status'])
                kwargs = f"{{'status': {status}}}"
            line = f"{str(event)} {kwargs}\n"
            fp.write(line)


def main(emitter):
    for event in iter(OptimizeEvent):
        emitter.on(event, create_handler(event))

    emitter.on(OptimizeEvent.WORKSPACE_INITIALIZED, save_workspace)
    atexit.register(store_events)


if current_process().name == "MainProcess":
    emitter = default_event_emitter()
    events = []
    workspace = None

    main(emitter)

```

You can import it in any functional test and events will be recorded for that test.
