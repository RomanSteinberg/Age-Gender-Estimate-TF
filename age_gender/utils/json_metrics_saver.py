import json
import numpy as np
from collections import deque
from pathlib import Path


class JsonMetricsWriter:
    def __init__(self, file_name, metrics_and_errors, val_frequency):
        self.file_name = file_name
        self.metrics_and_errors = metrics_and_errors
        self.val_frequency = val_frequency
        self.metrics_list = list()

    def create_metric_deque(self):
        metrics_deque = dict()
        for name in self.metrics_and_errors:
            metrics_deque[name] = deque(maxlen=self.val_frequency)
        return metrics_deque

    def dump(self, batch, files, deque):
        current_metrics = {
            'batch': batch,
            'files': [s.decode('utf-8') for s in files],
            'mae_deque': [float(n) for n in deque['mae']],
            'accuracy_deque': [float(n) for n in deque['gender_acc']],
            'mae': float(np.mean(deque['mae'])),
            'gender_accuracy': float(np.mean(deque['gender_acc'])),
        }
        self.metrics_list.append(current_metrics)
        json.dump(self.metrics_list, Path(self.file_name).open(mode='w'))

