# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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


import contextlib


class UserError(Exception):
    pass


class UserErrorContext(contextlib.AbstractContextManager):
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            return
        raise UserError(exc_value)


class TensorTypeError(TypeError):
    pass
