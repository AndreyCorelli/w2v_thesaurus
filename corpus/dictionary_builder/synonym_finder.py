from typing import List, Dict, Tuple

from corpus.models import WordCard


class SynonymFinder:
    def __init__(self,
                 lang_a_cards: Dict[str, WordCard],
                 lang_b_cards: Dict[str, WordCard]):
        self.lang_a_cards = lang_a_cards
        self.lang_b_cards = lang_b_cards
        self.lang_a_vectors: Dict[str, Tuple[float, ...]] = {}
        self.lang_a_vectors: Dict[str, Tuple[float, ...]] = {}
        self.vector_distance_cache: Dict[str, float] = {}

    def pre_calculate(self):
        self._build_vectors()
        # find neighbours for each word

    def find_synonyms(self,
                      a_to_b: bool = True,
                      wrd: str = '',
                      synonym_count: int = 5,
                      neighbours_count: int = 5):
        pass

    def _build_vectors(self):
        self.lang_a_vectors = {w: self._build_word_vector(w) for w in self.lang_a_cards.values()}
        self.lang_b_vectors = {w: self._build_word_vector(w) for w in self.lang_b_cards.values()}
        # normalize vectors: let each coordinate varies in [0..1]
        self._normalize_vectors(self.lang_a_vectors)
        self._normalize_vectors(self.lang_b_vectors)

    @classmethod
    def _build_word_vector(cls, card: WordCard) -> Tuple[float, ...]:
        return card.vector_length, card.vector_variance, card.frequency, card.frequency_rel_rank, card.non_uniformity

    @classmethod
    def _normalize_vectors(cls, lang_vectors: Dict[str, Tuple[float, ...]]):
        first_v = lang_vectors[next(iter(lang_vectors))]
        # find min and max for each column
        min_max = [(v, v, 0) for v in first_v]
        for v in lang_vectors.values():
            for i in range(len(v)):
                c = v[i]
                mi, ma, _ = min_max[i]
                min_max[i] = min(mi, c), max(ma, c), 0
        min_max = [(mi, ma, ma - mi if ma != mi else 1) for mi, ma, _r in min_max]

        # scale each value
        for wrd in lang_vectors:
            v = lang_vectors[wrd]
            v = [(v[i] - min_max[i][0]) / min_max[i][2] for i in range(len(min_max))]
            lang_vectors[wrd] = v
