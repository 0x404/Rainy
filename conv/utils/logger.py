import logging
from logging.config import fileConfig


def get_logger(name):
    fileConfig("configs/logger.conf")
    return logging.getLogger(name)
