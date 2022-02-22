# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from typing import Callable

from torch.utils.data import DataLoader  # pytype: disable=import-error
from transformers import DataCollatorWithPadding


def get_default_preprocess_function(dataset_name, tokenizer, max_sequence_length):
    def _preprocess_text_dataset(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_sequence_length)

    def _preprocess_qa_dataset(examples):
        return tokenizer(examples["question"], examples["context"], truncation=True, max_length=max_sequence_length)

    return {"imdb": _preprocess_text_dataset, "squad": _preprocess_qa_dataset}[dataset_name]


class HFDataLoaderFactory:
    def __init__(self, dataset, tokenizer, preprocess_function, onnx_config, device):

        self._inputs = set(onnx_config.inputs.keys())
        self._data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.device = device

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        self._dataset = tokenized_dataset.remove_columns(
            [c for c in tokenized_dataset.column_names if c not in self._inputs]
        )

    def __call__(self, batch_size: int = 1) -> Callable:
        dataloader_ = DataLoader(
            self._dataset,
            batch_size=batch_size,
            collate_fn=self._data_collator,
        )

        def dataloader():
            for batch in dataloader_:
                yield {**batch}

        return dataloader
