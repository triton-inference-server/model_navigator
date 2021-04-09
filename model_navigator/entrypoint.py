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
import signal
import sys

from model_navigator.optimizer.config import OptimizerConfig
from .cli import CLI
from .config import ModelNavigatorConfig
from .log import set_logger, log_dict
from .model_navigator import ModelNavigator
from .model_navigator_exceptions import ModelNavigatorException

MAX_NUMBER_OF_INTERRUPTS = 3

LOGGER = logging.getLogger(__name__)


def interrupt_handler(signal, frame):
    """
    A signal handler to properly
    shutdown the model navigator on
    interrupt
    """

    global exiting
    exiting += 1
    logging.info(f"Received SIGINT. Exiting ({exiting}/{MAX_NUMBER_OF_INTERRUPTS})...")

    if exiting == MAX_NUMBER_OF_INTERRUPTS:
        sys.exit(1)
    return


def main():
    """
    Main entrypoint of model_navigator
    """

    global exiting

    # Number of Times User Requested Exit
    exiting = 0

    signal.signal(signal.SIGINT, interrupt_handler)

    try:
        config = ModelNavigatorConfig().merge(OptimizerConfig())
        cli = CLI(config)
        cli.parse()
    except ModelNavigatorException as e:
        logging.error(f"Model Navigator encountered an error: {e}")
        sys.exit(1)

    set_logger(verbose=config.verbose)

    log_dict("Model Navigator config:", config.get_all_config())

    try:
        # Only check for exit after the events that take a long time.
        if exiting:
            return

        model_navigator = ModelNavigator(config)

        if exiting:
            return

        LOGGER.info("Starting Model Navigator")
        model_navigator.run()
        LOGGER.info("done")
    except ModelNavigatorException as e:
        LOGGER.exception(f"Model Navigator encountered an error: {e}")


if __name__ == "__main__":
    main()
