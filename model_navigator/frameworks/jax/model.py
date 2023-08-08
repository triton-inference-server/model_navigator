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
"""JAX utils."""

from typing import Any


class JaxModel:
    """Wrapper on a JAX infer function and params."""

    def __init__(self, model: Any, params: Any) -> None:
        """Initialize JaxModel.

        Args:
            model (Any): Inference function.
            params (Any): JAX Parameters.
        """
        self._model = model
        self._params = params

    @property
    def model(self) -> Any:
        """JAX inference function.

        Returns:
            Any: JAX inference function.
        """
        return self._model

    @property
    def params(self):
        """JAX parameters.

        Returns:
            Any: JAX parameters.
        """
        return self._params

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the inference on the model.

        Returns:
            Any: Model inference output.
        """
        return self._model(*args, **kwargs, params=self._params)
