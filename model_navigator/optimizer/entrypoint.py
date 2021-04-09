#!/usr/bin/env python3
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
import logging
import pickle
import sys
import traceback
from pathlib import Path

import sh

import model_navigator.optimizer.transformers  # noqa
from ..log import set_logger, set_tf_verbosity, dump_sh_logs, log_dict

LOGGER = logging.getLogger("optimizer.entrypoint")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optimizer run in docker entrypoint")
    parser.add_argument("job_spec_path", type=str)
    parser.add_argument("results_path", type=str)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    set_logger(verbose=args.verbose)
    set_tf_verbosity(verbose=args.verbose)
    if args.verbose:
        log_dict("args", vars(args))

    job_spec_path = Path(args.job_spec_path)
    results_path = Path(args.results_path)
    with job_spec_path.open("rb") as job_spec_file:
        job_spec = pickle.load(job_spec_file)
        pipeline, src_model, config = job_spec

    return_code = 0
    results = []
    try:
        for model in pipeline.execute(src_model, config=config):
            results.append(model)
    except sh.ErrorReturnCode as e:
        LOGGER.warning("Model Navigator encountered an error during model optimization")
        dump_sh_logs("stdout", e.stdout, limit=256)
        return_code = -1
    except Exception as e:
        LOGGER.warning(f"Model Navigator encountered an error during model optimization; {e}")
        LOGGER.warning(traceback.format_exc())
        return_code = -1
    finally:
        with results_path.open("wb") as results_file:
            pickle.dump(results, results_file)
        results_path.chmod(0o666)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
