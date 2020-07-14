from logging import getLogger, Formatter, DEBUG, handlers, StreamHandler
import os
from pathlib import Path


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
        formatter = Formatter(
            '%(asctime)s %(name)s.%(funcName)s +%(lineno)s: '
            '%(levelname)-8s [%(process)d] %(message)s'
        )
        logger = getLogger('session_log')
        logger.setLevel(DEBUG)
        file_title = os.path.join(os.getcwd(), 'logs', 'session.log')
        if 'logs' not in os.listdir():
            file_title = f'{os.path.join(os.getcwd(), file_title)}'
            Path(file_title).parent.mkdir(parents=True, exist_ok=True)
        file_handler = handlers.RotatingFileHandler(file_title, backupCount=10)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
