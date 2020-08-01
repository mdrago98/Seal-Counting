from pathlib import Path
from time import perf_counter

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame

from yolov3_tf2.gpu_monitor import Monitor


class EpochTimer(keras.callbacks.Callback):
    """
    A callback that times the epochs
    """

    def __init__(self, output, delay=10):
        super().__init__()
        self.times = []
        self.output = output
        self.delay = delay
        self.timetaken = perf_counter()
        self.initial_usage = 0
        self.monitor = Monitor(self.delay)

    def on_train_begin(self, logs=None):
        self.monitor.start()

    def on_predict_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.times.append((epoch, perf_counter() - self.timetaken))
        self.epoch_gpu_rates += [(epoch, self.res.memory)]

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        self.monitor.stop()
        df = DataFrame(self.monitor.results, columns=["Reading", "Memory Usage", "GPU Load"])
        df.to_csv(self.output)
