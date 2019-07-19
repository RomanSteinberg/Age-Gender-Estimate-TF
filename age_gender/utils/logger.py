# -*- coding: utf-8 -*-
import logging
import os
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from pythonjsonlogger import jsonlogger


class MetricFormatter(jsonlogger.JsonFormatter):
    def __init__(self):
        super(MetricFormatter, self).__init__('(asctime) (levelname) (message)')


class Logger:
    def __init__(self, logger_name, logs_folder):
        self.logs_folder = logs_folder
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel('DEBUG')
        handlers = list()
        logging.raiseExceptions = True

        if not os.path.exists(self.logs_folder):
            os.mkdir(self.logs_folder)
        data_handler = logging.FileHandler(os.path.join(self.logs_folder, f'{logger_name}.log'))
        data_handler.setFormatter(MetricFormatter())
        handlers.append(data_handler)

        # организуем неблокирующие записи логов
        logs_queue = Queue(-1)
        queue_handler = QueueHandler(logs_queue)
        self.listener = QueueListener(logs_queue, *handlers)
        self.logger.addHandler(queue_handler)
        self.listener.start()

    def __del__(self):
        self.listener.stop()

    def debug(self, message, extra=None):
        self.logger.debug(message, extra=extra)

    def info(self, message, extra=None):
        self.logger.info(message, extra=extra)

    def warning(self, message, extra=None):
        self.logger.warning(message, extra=extra)

    def error(self, message, extra=None):
        self.logger.error(message, extra=extra)

    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)
