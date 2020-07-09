from logging import handlers, getLogger, Formatter, DEBUG, StreamHandler, Logger
from os import path, listdir


def get_logger() -> Logger:
    """
    Initialises a new logger.
    :return: the logger instance
    """
    formatter = Formatter(
        '%(asctime)s %(name)s.%(funcName)s +%(lineno)s: '
        '%(levelname)-8s [%(process)d] %(message)s'
    )
    logger = getLogger('session_log')
    logger.setLevel(DEBUG)
    file_title = path.join('Logs', 'session.log')
    if 'Logs' not in listdir():
        file_title = f'{path.join("..", file_title)}'
    file_handler = handlers.RotatingFileHandler(file_title, backupCount=10)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
