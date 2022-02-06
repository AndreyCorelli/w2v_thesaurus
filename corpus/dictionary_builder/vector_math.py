import math
from typing import List


def get_cosine_distance(a: List[float], b: List[float]):
    ma, mb, ab = 0, 0, 0
    for i in range(len(a)):
        ab += a[i] * b[i]
        ma += a[i] * a[i]
        mb += b[i] * b[i]

    denom = math.sqrt(ma) * math.sqrt(mb)
    return abs(ab) / denom if denom > 0 else 0
