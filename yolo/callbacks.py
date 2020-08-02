from pathlib import Path
from time import perf_counter, time

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame

from yolo.gpu_monitor import Monitor


class GPUReport(keras.callbacks.Callback):
    """
    A callback monitors the gpu memory and utilisation.
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

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        self.monitor.stop()
        df = DataFrame(self.monitor.results, columns=["Reading", "Memory Usage", "GPU Load"])
        df.to_csv(self.output)


class TimeHistory(keras.callbacks.Callback):
    """
    Monitors the time it takes to process an epoch
    """

    def __init__(self, output):
        super().__init__()
        self.times = []
        self.seen_sampes = 0
        self.epoch_time_start = None
        self.output = output

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}

    def on_predict_end(self, logs=None):
        if logs is None:
            logs = {}
        self.seen_sampes += 1

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epoch_time_start = perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.times.append((epoch, perf_counter() - self.epoch_time_start, self.seen_sampes))

    def on_train_end(self, logs=None):
        df = DataFrame(self.times, columns=["Epoch", "Time", "Seen Samples"])
        df.to_csv(self.output)
