from logging import getLogger, Formatter, DEBUG, handlers, StreamHandler
import os
from pathlib import Path

from helpers import default_logger


class LoggerMixin(object):
    """
    A mix in that adds logging capabilities to the class exposed as logger
    """

    @property
    def logger(self):
        """
        Defines the logging property
        :return: the initialised logger
        """
        return default_logger
