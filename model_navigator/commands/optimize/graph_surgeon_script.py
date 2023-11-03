# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Graph Surgeon ONNX optimization script."""

import pathlib
import shutil
import tempfile

import fire
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants

from model_navigator.core.logger import LOGGER


class _Optimizer:
    def __init__(self, onnx_graph):
        self.graph = gs.import_onnx(onnx_graph)

    def cleanup(self):
        LOGGER.info("Cleaning up ONNX graph")
        self.graph.cleanup().toposort()

    def select_outputs(self, keep, names=None):
        LOGGER.info(f"Selecting outputs: {keep}")
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        LOGGER.info("Folding constants")
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)

    def infer_shapes(self):
        LOGGER.info("Inferring shapes")
        onnx_graph = gs.export_onnx(self.graph)
        if self.exceeds_protobuf_limit(onnx_graph):
            LOGGER.warn("Model size exceeds supported 2GB limit, unable to infer shapes.")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)
        self.graph = gs.import_onnx(onnx_graph)

    def get_graph(self):
        return gs.export_onnx(self.graph)

    def save(self, path: str):
        """Saves ONNX graph to file.

        Args:
            path (str): Path to save ONNX graph to.
        """
        LOGGER.info(f"Saving ONNX graph to: {path}")

        temp_dir = pathlib.Path(tempfile.mkdtemp())
        target_path = pathlib.Path(path)

        graph = self.get_graph()
        external_weights = self.exceeds_protobuf_limit(graph)
        onnx.save(graph, str(temp_dir / target_path.name), save_as_external_data=external_weights)

        for file in target_path.parent.iterdir():
            if not file.name.endswith((".py", ".sh")):
                file.unlink()

        for onnx_file in temp_dir.iterdir():
            shutil.move(str(onnx_file), str(target_path.parent / onnx_file.name))

        shutil.rmtree(temp_dir)

    def exceeds_protobuf_limit(self, graph):
        return graph.ByteSize() > 2**31


def optimize(onnx_path: str):
    """Optimize ONNX model using GraphSurgeon.

    The following optimizations are performed:
    - Fold constants
    - Infer shapes
    - Cleanup

    Args:
        onnx_path (str): Path to ONNX model.
    """
    onnx_graph = onnx.load(onnx_path)
    opt = _Optimizer(onnx_graph)
    opt.cleanup()
    opt.fold_constants()
    opt.infer_shapes()
    opt.cleanup()
    opt.save(onnx_path)


if __name__ == "__main__":
    fire.Fire(optimize)
