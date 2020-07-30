from time import perf_counter

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt


class EpochTimer(keras.callbacks.Callback):
    """
    A callback that times the epochs 
    """

    def __init__(self):
        super().__init__()
        self.times = []
        self.timetaken = perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.times.append((epoch, perf_counter() - self.timetaken))

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        plt.xlabel("Epoch")
        plt.ylabel("Total time taken until an epoch in seconds")
        plt.plot(*zip(*self.times))
        plt.show()
