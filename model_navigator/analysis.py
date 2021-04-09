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
from typing import Dict, List

from dataclasses import dataclass


@dataclass
class Analysis:
    model_name: str
    model_config_path: str
    engine_count: int
    preferred_batch_sizes: List[int]

    @classmethod
    def from_analyzer_result(cls, result: Dict):
        model_name = result["Model"]
        model_config_path = result["Model Config Path"]
        res_engine_count = result["Instance Group"]
        engine_count, _ = res_engine_count.split("/")
        preferred_batch_sizes = result["Dynamic Batcher Sizes"].strip("[]")
        preferred_batch_sizes = list(map(int, preferred_batch_sizes.split(" ")))

        analysis = Analysis(
            model_name=model_name,
            model_config_path=model_config_path,
            engine_count=engine_count,
            preferred_batch_sizes=preferred_batch_sizes,
        )

        return analysis
