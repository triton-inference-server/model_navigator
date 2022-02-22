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

from enum import Enum

# pytype: disable=import-error
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)

# pytype: enable=import-error


class Task(Enum):
    BASE = "base"
    SEQUENCE_CLASSIFICATION = "sequence-classification"
    QUESTION_ANSWERING = "question-answering"
    FILL_MASK = "fill-mask"
    TEXT_GENERATION = "text-generation"


_TASK_TO_AUTOMODEL = {
    Task.BASE: AutoModel,
    Task.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    Task.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    Task.FILL_MASK: AutoModelForMaskedLM,
    Task.TEXT_GENERATION: AutoModelForCausalLM,
}


def get_automodel(task: Task):

    return _TASK_TO_AUTOMODEL[task]
