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
"""Script for exporting Keras model to SavedModel."""

import pathlib
from typing import Any, Dict, Optional

import fire
import tensorflow as tf  # pytype: disable=import-error

from model_navigator.core.tensor import TensorMetadata


def get_model() -> tf.keras.Model:
    """Get model instance.

    Returns:
        Model to be exported.
    """
    raise NotImplementedError("Please implement the get_model() function to reproduce the export error.")


def export(
    exported_model_path: str,
    input_metadata: Dict[str, Any],
    output_metadata: Dict[str, Any],
    navigator_workspace: Optional[str] = None,
) -> None:
    """Export Keras model to SavedModel.

    Args:
        exported_model_path (str): Path to SavedModel relative to navigator_workspace path.
        input_metadata (Dict[str, Any]): Input metadata.
        output_metadata (Dict[str, Any]): Output metadata.
        navigator_workspace (Optional[str], optional): Model Navigator workspace path.
            If None use current workdir path. Defaults to None.
    """
    model = get_model()

    input_metadata = TensorMetadata.from_json(input_metadata)
    output_metadata = TensorMetadata.from_json(output_metadata)

    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    @tf.function()
    def predict(inputs_dict):
        inputs = input_metadata.unflatten_sample(inputs_dict)
        outputs = model(inputs)
        return output_metadata.flatten_sample(outputs)

    input_specs = {
        name: tf.TensorSpec(shape=[d if d != -1 else None for d in spec.shape], dtype=spec.dtype, name=name)
        for name, spec in input_metadata.items()
    }
    signatures = predict.get_concrete_function(input_specs)

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = navigator_workspace / exported_model_path

    tf.keras.models.save_model(  # pytype: disable=module-attr
        model=model, filepath=exported_model_path.as_posix(), overwrite=True, signatures=signatures
    )


if __name__ == "__main__":
    fire.Fire(export)
