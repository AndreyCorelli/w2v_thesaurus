from typing import List, Tuple


class SortedItems:
    def __init__(self, max_count: int):
        self.max_count = max_count
        self.items: List[Tuple[int, float]] = []
        self.max: float = 0

    def update(self, val: float, id: int):
        if len(self.items) < self.max_count:
            self.items.append((id, val))
            self.max = max(val, self.max)
            self.items.sort(key=lambda it: it[1])
            return

        if val >= self.max:
            return

        self.items[-1] = (id, val)
        self.items.sort(key=lambda it: it[1])
        self.max = self.items[-1][1]