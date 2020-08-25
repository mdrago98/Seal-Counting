from threading import Thread
import time
import GPUtil as gpu


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.results = []
        self.gpu = gpu.getGPUs()[0]

    def run(self):
        reading = 0
        while not self.stopped:
            self.results += [
                (
                    reading,
                    gpu.getGPUs()[0].memoryUtil,
                    gpu.getGPUs()[0].memoryTotal,
                    gpu.getGPUs()[0].memoryUsed,
                    gpu.getGPUs()[0].load,
                )
            ]
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
