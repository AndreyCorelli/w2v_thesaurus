from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class WordCard:
    lang_code: str = None
    word: str = None
    stem: str = None
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

    """
    def __init__(self, w: Optional['WordCard']):
        super().__init__()
        if not w:
            return
        self.word = w.word
        self.root = w.root
        self.lang_code = w.lang_code
        self.frequency = w.frequency
        self.frequency_rank = w.frequency_rank
        self.non_uniformity = w.non_uniformity
        self.vector_length = w.vector_length
        self.vector_variance = w.vector_variance
        self.rel_length = w.rel_length
        self.prob_repeats = w.prob_repeats

        self.vector = w.vector
        self.neighbours = w.neighbours
    """

    def __str__(self):
        word = f'[{self.word}]' if not self.stem or self.stem == self.word \
            else f'[{self.word}] ({self.stem})'
        return f'{word}, f={self.frequency:.3f}, rank={self.frequency_rank}, u={self.non_uniformity:.2f}, ' + \
               f'Vl={self.vector_length:.2f}, Vv={self.vector_variance:.2f}, len={self.rel_length:.2f}'

    def __repr__(self):
        return str(self)

    def is_stem(self) -> bool:
        return self.word == self.stem
