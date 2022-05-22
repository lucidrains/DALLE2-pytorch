import time

# time helpers

class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_time = time.time()

    def elapsed(self):
        return time.time() - self.last_time

# print helpers

def print_ribbon(s, symbol = '=', repeat = 40):
    flank = symbol * repeat
    return f'{flank} {s} {flank}'
