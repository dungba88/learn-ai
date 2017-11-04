import random
import numpy as np

class MiniBatchGenerator(object):

    def __init__(self, max_size: int, x_mapper: callable, y_mapper: callable):
        self.arr = [idx for idx in range(max_size)]
        self.max_size = max_size
        self.x_mapper = x_mapper
        self.y_mapper = y_mapper
        random.shuffle(self.arr)
        self.train_idx = 0
        self.test_idx = 0
        self.train_arr = self.arr[:]
        self.test_arr = []
        self.reset()

    def split_train_test(self, test_size=0.2):
        train_size = int(self.max_size * (1 - test_size))
        self.train_arr = self.arr[:train_size]
        self.test_arr = self.arr[train_size:]
        self.reset()

    def load_next_train_batch(self, batch_size: int):
        batch = self.__next_batch(batch_size, self.train_idx, self.train_arr)
        self.train_idx += len(batch)
        return self.__load(batch)

    def load_next_test_batch(self, batch_size: int):
        batch = self.__next_batch(batch_size, self.test_idx, self.test_arr)
        self.test_idx += len(batch)
        return self.__load(batch)

    def reset(self):
        self.train_idx = 0
        self.test_idx = 0
        random.shuffle(self.train_arr)
        random.shuffle(self.test_arr)

    def __next_batch(self, batch_size, first_idx, arr):
        remaining = len(arr) - first_idx
        if remaining == 0:
            return []
        if batch_size > remaining:
            batch_size = remaining
        result = arr[first_idx:first_idx + batch_size]
        return result

    def __load(self, batch):
        if not batch:
            return None, None
        X = np.array([self.x_mapper(item) for item in batch])
        y = np.array([self.y_mapper(item) for item in batch])
        return X, y
