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

from functools import total_ordering

from model_navigator.exceptions import ModelNavigatorException
from model_navigator.record import RecordType


@total_ordering
class Measurement:
    """
    Encapsulates the set of metrics obtained from a single
    perf_analyzer run
    """

    def __init__(self, result, comparator):
        """
        result : dict with result info
        """
        self.result = result
        self._comparator = comparator
        self._records = RecordType.get_all_record_types()

    def get_value_of_metric(self, tag):
        """
        Parameters
        ----------
        tag : str
            A human readable tag that corresponds
            to a particular metric

        Returns
        -------
        Record
            metric Record corresponding to
            the tag, in this measurement
        """

        if tag in self._records:
            cls = self._records[tag]
            header = cls.header()
            value = float(self.result[header])
            metric = cls(value=value)
            return metric
        else:
            raise ModelNavigatorException(f"No metric corresponding to tag {tag}" " found in measurement")

    def __eq__(self, other):
        """
        Check whether two sets of measurements are equivalent
        """

        return self._comparator.compare_measurements(self, other) == 0

    def __lt__(self, other):
        """
        Checks whether a measurement is better than
        another

        If True, this means this measurement is better
        than the other.
        """

        # seems like this should be == -1 but we're using a min heap
        return self._comparator.compare_measurements(self, other) == 1
