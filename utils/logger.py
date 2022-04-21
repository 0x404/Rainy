"""Logger utils"""
import logging
from logging.config import fileConfig


def get_logger(name):
    """Get logger"""
    fileConfig("configs/logger.conf")
    return logging.getLogger(name)
