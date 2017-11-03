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

    def load_next_batch(self, batch_size: int, X_mapper: callable, y_mapper: callable):
        batch = self.next_batch(batch_size)
        if not batch:
            return None, None
        return np.array([X_mapper(item) for item in batch]), np.array([y_mapper(item) for item in batch])

    def next_batch(self, batch_size: int):
        remaining = len(self.arr) - self.idx
        if remaining == 0:
            return None
        if batch_size > remaining:
            batch_size = remaining
        result = self.arr[self.idx:self.idx + batch_size]
        self.idx += batch_size
        return result
