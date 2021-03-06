from typing import List
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class WordCard:
    word: str = None
    # stem: str = None
    frequency: float = 0
    frequency_rank: int = 0
    frequency_rel_rank: float = 0
    non_uniformity: float = 0
    vector_length: float = 0
    vector_variance: float = 0
    vector: List[float] = None
    rel_length: float = 0
    prob_repeats: float = 0
    neighbours: List[str] = None

    def __str__(self):
        return f'{self.word}, f={self.frequency:.3f}, rank={self.frequency_rank}, u={self.non_uniformity:.2f}, ' + \
               f'Vl={self.vector_length:.2f}, Vv={self.vector_variance:.2f}, len={self.rel_length:.2f}'

    def __repr__(self):
        return str(self)

    # def is_stem(self) -> bool:
    #     return self.word == self.stem
