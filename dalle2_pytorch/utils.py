import time

class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_time = time.time()

    def elapsed(self):
        return time.time() - self.last_time
