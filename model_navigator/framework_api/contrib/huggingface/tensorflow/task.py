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

from collections import OrderedDict
from enum import Enum

# pytype: disable=import-error
from transformers import PreTrainedModel
from transformers.models.auto.modeling_tf_auto import (
    TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    TF_MODEL_MAPPING_NAMES,
)

# pytype: enable=import-error


class Task(Enum):
    SEQUENCE_CLASSIFICATION = "sequence-classification"
    QUESTION_ANSWERING = "question-answering"
    CAUSAL_LM = "causal-lm"
    MASKED_LM = "masked-lm"
    BASE = "base"


TASK_MODELS_MAPPING = OrderedDict(
    [
        (Task.SEQUENCE_CLASSIFICATION, set(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values())),
        (Task.QUESTION_ANSWERING, set(TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES.values())),
        (Task.CAUSAL_LM, set(TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values())),
        (Task.MASKED_LM, set(TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES.values())),
        (Task.BASE, set(TF_MODEL_MAPPING_NAMES.values())),
    ]
)

TASK_OUTPUTS_MAPPING = OrderedDict(
    [
        (Task.SEQUENCE_CLASSIFICATION, OrderedDict([("logits", {0: "batch"})])),
        (
            Task.QUESTION_ANSWERING,
            OrderedDict([("start_logits", {0: "batch", 1: "sequence"}), ("end_logits", {0: "batch", 1: "sequence"})]),
        ),
        (Task.CAUSAL_LM, OrderedDict([("logits", {0: "batch", 1: "sequence"})])),
        (Task.MASKED_LM, OrderedDict([("logits", {0: "batch", 1: "sequence"})])),
    ]
)


def get_task_from_model(model: PreTrainedModel) -> Task:
    for task, models in TASK_MODELS_MAPPING.items():
        if model.__class__.__name__ in models:
            return task
    raise ValueError(f"Taks for {model.__class__.__name__} is unknown")
