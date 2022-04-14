#!/usr/bin/env python3
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

import argparse
import tempfile
from pathlib import Path

import torch  # pytype: disable=import-error
from modeling import BertConfig, BertForQuestionAnswering  # pytype: disable=import-error
from run_squad import convert_examples_to_features, read_squad_examples  # pytype: disable=import-error
from tokenization import BertTokenizer  # pytype: disable=import-error
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset  # pytype: disable=import-error

import model_navigator as nav

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None, type=str, required=True, help="The BERT model config")
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--vocab_file", type=str, default=None, required=True, help="Vocabulary mapping/file BERT was pretrainined on"
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:

        navigator_workdir = Path(args.workdir)
        download_dir = Path(tmp_dir) / "download"
        download_dir.mkdir(exist_ok=True, parents=True)

        config = BertConfig.from_json_file(args.config_file)
        # Padding for divisibility by 8
        if config.vocab_size % 8 != 0:
            config.vocab_size += 8 - (config.vocab_size % 8)

        model = BertForQuestionAnswering(config)

        tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512)

        eval_examples = read_squad_examples(
            input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative
        )
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False,
        )

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        # all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)  # , all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        pkg_desc = nav.torch.export(
            model=model,
            model_name="bert_pyt",
            workdir=navigator_workdir,
            dataloader=eval_dataloader,
            sample_count=1,
            input_names=("input_ids", "token_type_ids", "attention_mask"),
        )
        expected_formats = ()  # ("torchscript-script", "torchscript-trace", "onnx", "trt-fp32", "trt-fp16") # fails on the pytorch:22.02 container
        for format, runtimes_status in pkg_desc.get_formats_status().items():
            for runtime, status in runtimes_status.items():
                assert (status == nav.Status.OK) == (
                    format in expected_formats
                ), f"{format} {runtime} status is {status}, but expected formats are {expected_formats}."
        pkg_desc.save(navigator_workdir / "bert_pyt.nav")

    nav.LOGGER.info("All models passed.")
