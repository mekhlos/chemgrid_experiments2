from typing import Optional

import numpy as np


class PriorityQueue:
    def __init__(self, max_len: Optional[int] = None):
        self.values = []
        self.priorities = []
        self.max_len = max_len
        self.rng = np.random.default_rng(seed=0)

    def add(self, value, priority=0):
        i = 0
        for p in self.priorities:
            if priority > p:
                break

            i += 1

        self.values.insert(i, value)
        self.priorities.insert(i, priority)

        if self.max_len is not None and len(self) > self.max_len:
            self.values.pop()
            self.priorities.pop()

    def get_probs(self) -> np.ndarray:
        return self.priorities / np.sum(self.priorities)

    def sample(self):
        i = self.rng.choice(len(self), p=self.get_probs())
        return self[i]

    def pop(self):
        v, p = self.values.pop(0), self.priorities.pop(0)
        return v

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self) -> int:
        return len(self.values)


if __name__ == '__main__':
    pq = PriorityQueue(5)
    pq.add("A", 26)
    pq.add("B", 20)
    pq.add("C", 27)
    pq.add("D", 28)
    pq.add("E", 28)

    print(pq.values)
    print(pq.priorities)
