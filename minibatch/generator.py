import random
import numpy as np

class MiniBatchGenerator(object):

    def __init__(self, max_size: int):
        self.arr = [idx for idx in range(max_size)]
        self.idx = 0
        random.shuffle(self.arr)

    def reset(self):
        self.idx = 0
        random.shuffle(self.arr)

    def load_next_batch(self, file_dict, batch_size):
        return self.load_batch(self.next_batch(batch_size), file_dict)

    def next_batch(self, batch_size):
        remaining = len(self.arr) - self.idx
        if remaining == 0:
            return None
        if batch_size > remaining:
            batch_size = remaining
        result = self.arr[self.idx:self.idx + batch_size]
        self.idx += batch_size
        return result

    def load_batch(self, batch, file_dict: dict):
        if not batch:
            return None
        data = np.array([np.load(file_dict[idx]) for idx in batch])
        return data, batch
