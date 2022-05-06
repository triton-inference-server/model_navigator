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
import json
import pathlib
import subprocess
import sys
import tempfile

import torch  # pytype: disable=import-error

import model_navigator as nav

MODEL_NAMES = ["BART"]
REQUIREMENTS = ["tokenizers"]

ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."


def setup_env(workdir):
    cmd = [
        "git",
        "clone",
        "https://github.com/NVIDIA/DeepLearningExamples",
        (workdir / "DeepLearningExamples").as_posix(),
    ]
    subprocess.run(cmd, check=True)
    bart_path = workdir / "DeepLearningExamples/PyTorch/LanguageModeling/BART/"
    sys.path.append(bart_path.as_posix())
    cmd = [sys.executable, "-m", "pip", "install"] + REQUIREMENTS
    subprocess.run(cmd, check=True)

    return bart_path


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
        "--model-path", type=str, help="like facebook/bart-large-cnn or path to ckpt", default="facebook/bart-large"
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--num_return_sequences", type=int, default=1, required=False, help="How many sequences to return"
    )
    parser.add_argument("--eval_max_gen_length", type=int, default=142, help="never generate more than n tokens")
    parser.add_argument(
        "--eval_beams",
        type=int,
        default=4,
        required=False,
        help="# beams to use. 0 corresponds to not using beam search.",
    )
    parser.add_argument(
        "--max_source_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=142,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")

    return parser.parse_args()


def main():

    args = parse_args()
    if args.list_models:
        print(MODEL_NAMES)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        bart_path = setup_env(pathlib.Path(tmpdir))
        from bart.configuration.configuration_bart import BartConfig
        from bart.modeling.modeling_bart import BartForConditionalGeneration
        from bart.tokenization.tokenization_bart import BartTokenizer
        from export_utils.generation_onnx import BARTBeamSearchGenerator

        config_file = bart_path / "configs/config.json"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        config = BartConfig(**json.load(open(config_file)))
        config.fp16 = args.fp16

        model = BartForConditionalGeneration.from_pretrained(args.model_path, config=config)
        tokenizer = BartTokenizer.from_pretrained(args.model_path)

        model.to(device)

        model.eval()
        bart_script_model = torch.jit.script(BARTBeamSearchGenerator(model))

        inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=args.max_source_length, return_tensors="pt").to(device)

        dataloader = [
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "num_beams": torch.tensor(args.eval_beams).to(device),
                "max_length": torch.tensor(args.eval_max_gen_length).to(device),
                "decoder_start_token_id": torch.tensor(model.config.decoder_start_token_id).to(device),
            }
        ]

        pkg_desc = nav.torch.export(
            model=bart_script_model,
            model_name=f"{args.model_name}_pyt",
            dataloader=dataloader,
            sample_count=1,
            batch_dim=None,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "output_ids": {0: "batch", 1: "seq_out"},
            },
            input_names=("input_ids", "attention_mask", "num_beams", "max_length", "decoder_start_token_id"),
            opset=14,
            target_device=device,
            atol=1e-3,
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
        pkg_desc.save(output_path, keep_workdir=args.keep_workdir, override=True)


if __name__ == "__main__":
    main()
