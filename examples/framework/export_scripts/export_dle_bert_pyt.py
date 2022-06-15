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
import os
import pathlib
import subprocess
import sys
import tempfile

import torch  # pytype: disable=import-error
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset  # pytype: disable=import-error

import model_navigator as nav

MODEL_NAMES = ["BERT"]


def setup_env(workdir):
    cmd = [
        "git",
        "clone",
        "https://github.com/NVIDIA/DeepLearningExamples",
        (workdir / "DeepLearningExamples").as_posix(),
    ]
    subprocess.run(cmd, check=True)
    bert_path = workdir / "DeepLearningExamples/PyTorch/LanguageModeling/BERT/"
    sys.path.append(bert_path.as_posix())
    bert_prep_working_dir = bert_path / "bert_prep"
    os.environ["BERT_PREP_WORKING_DIR"] = bert_prep_working_dir.as_posix()
    subprocess.run(
        [sys.executable, (bert_path / "data/bertPrep.py").as_posix(), "--action", "download", "--dataset", "squad"],
        check=True,
    )

    requirements = []
    with open(bert_path / "requirements.txt") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "onnx" in line or line.startswith("#"):
                continue
            requirements.append(line)
    cmd = [sys.executable, "-m", "pip", "install"] + requirements
    subprocess.run(cmd, check=True)

    return bert_path, bert_prep_working_dir


def get_verification_status_dummy(runner):
    """Dummy verification function."""
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name", type=str, choices=MODEL_NAMES)
    group.add_argument(
        "--list-models",
        action="store_true",
    )
    parser.add_argument(
        "--output-path",
        type=str,
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
    )
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
    return parser.parse_args()


def main():

    args = parse_args()
    if args.list_models:
        print(MODEL_NAMES)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        bert_path, bert_prep_working_dir = setup_env(pathlib.Path(tmpdir))
        from modeling import BertConfig, BertForQuestionAnswering  # pytype: disable=import-error
        from run_squad import convert_examples_to_features, read_squad_examples  # pytype: disable=import-error
        from tokenization import BertTokenizer  # pytype: disable=import-error

        config_file = bert_path / "bert_configs/base.json"
        predict_file = bert_prep_working_dir / "download/squad/v1.1/dev-v1.1.json"
        vocab_file = bert_path / "vocab/vocab"

        config = BertConfig.from_json_file(config_file)
        # Padding for divisibility by 8
        if config.vocab_size % 8 != 0:
            config.vocab_size += 8 - (config.vocab_size % 8)

        model = BertForQuestionAnswering(config)

        tokenizer = BertTokenizer(vocab_file, do_lower_case=args.do_lower_case, max_len=512)

        eval_examples = read_squad_examples(
            input_file=predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative
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
            model_name=f"{args.model_name}_pyt",
            dataloader=eval_dataloader,
            sample_count=1,
            input_names=("input_ids", "token_type_ids", "attention_mask"),
            opset=14,
        )

        for model_status in pkg_desc.navigator_status.model_status:
            for runtime_results in model_status.runtime_results:
                if runtime_results.status == nav.Status.OK:
                    runner = pkg_desc.get_runner(
                        format=model_status.format,
                        jit_type=model_status.torch_jit,
                        precision=model_status.precision,
                        runtime=runtime_results.runtime,
                    )
                    verified = get_verification_status_dummy(runner)
                    if verified:
                        pkg_desc.set_verified(
                            format=model_status.format,
                            jit_type=model_status.torch_jit,
                            precision=model_status.precision,
                            runtime=runtime_results.runtime,
                        )
                        nav.LOGGER.info(
                            f"{model_status.format=}, {model_status.torch_jit=}, {model_status.precision=}, {runtime_results.runtime=} verified."
                        )
                    else:
                        nav.LOGGER.warning(
                            f"{model_status.format=}, {model_status.torch_jit=}, {model_status.precision=}, {runtime_results.runtime=} not verified."
                        )

        output_path = args.output_path or f"{args.model_name}_pyt.nav"
        nav.save(pkg_desc, output_path, keep_workdir=args.keep_workdir, override=True)


if __name__ == "__main__":
    main()
