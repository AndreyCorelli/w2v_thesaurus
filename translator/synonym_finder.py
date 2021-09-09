from typing import List, Dict, Tuple
import numpy as np

from corpus.dictionary_builder.sorted_items import SortedItems
from corpus.models import WordCard


class SynonymsFound:
    def __init__(self):
        self.synonyms: List[str] = []
        self.message = ''

    def __repr__(self):
        return self.message if self.message else ', '.join([f'{s}' for s in self.synonyms])

    def __str__(self):
        return self.__repr__()


class SynonymFinder:
    def __init__(self,
                 lang_a_cards: List[WordCard],
                 lang_b_cards: List[WordCard]):
        self.lang_a_cards = lang_a_cards
        self.lang_b_cards = lang_b_cards
        self.a_index_by_word = {lang_a_cards[i].word: i for i in range(len(lang_a_cards))}
        self.b_index_by_word = {lang_b_cards[i].word: i for i in range(len(lang_b_cards))}
        self.a_index_by_id = {lang_a_cards[i].id: i for i in range(len(lang_a_cards))}
        self.b_index_by_id = {lang_b_cards[i].id: i for i in range(len(lang_b_cards))}

        self.card_a_by_word: Dict[str, WordCard] = {w.word: w for w in lang_a_cards}
        self.card_b_by_word: Dict[str, WordCard] = {w.word: w for w in lang_b_cards}
        self.card_a_by_id: Dict[int, WordCard] = {w.id: w for w in lang_a_cards}
        self.card_b_by_id: Dict[int, WordCard] = {w.id: w for w in lang_b_cards}
        self.lang_a_vectors: List[Tuple[float, ...]] = []
        self.lang_b_vectors: List[Tuple[float, ...]] = []
        self._build_vectors()

    def find_synonyms(self,
                      a_to_b: bool = True,
                      wrd: str = '',
                      synonym_count: int = 20) -> SynonymsFound:
        result = SynonymsFound()
        src_idx = self.a_index_by_word.get(wrd) if a_to_b else self.b_index_by_word.get(wrd)
        if src_idx < 0:
            result.message = 'Source word not found'
            return result

        src_card = self.lang_a_cards[src_idx] if a_to_b else self.lang_b_cards[src_idx]
        src_vectors = self.lang_a_vectors if a_to_b else self.lang_b_vectors
        dst_vectors = self.lang_b_vectors if a_to_b else self.lang_a_vectors

        # find <synonym_count> closest vectors in dst_vectors
        candidate_ids = self._find_closest_n_vectors(src_vectors[src_idx], dst_vectors, synonym_count)
        # get neighbours of the src item

        src_index_by_id = self.a_index_by_id if a_to_b else self.b_index_by_id
        src_neihgbour_vects = [src_vectors[src_index_by_id[id]] for id in src_card.neighbours]

        candidate_dist: List[Tuple[int, float]] = []
        for cd_id in candidate_ids:
            dst_nb_dist = self._get_distance_between_neighbours(src_neihgbour_vects, cd_id, a_to_b)
            candidate_dist.append((cd_id, dst_nb_dist))

        candidate_dist.sort(key=lambda cd: cd[1])
        dst_cards = self.lang_b_cards if a_to_b else self.lang_a_cards
        dst_words = [dst_cards[id].word for id, _ in candidate_dist]
        result.synonyms = dst_words
        return result

    def _get_distance_between_neighbours(self,
                                         src_neihgbour_vects: List[Tuple[float, ...]],
                                         dst_id: int,
                                         a_to_b: bool) -> float:
        dst_card = self.lang_b_cards[dst_id] if a_to_b else self.lang_a_cards[dst_id]
        dst_vectors = self.lang_b_vectors if a_to_b else self.lang_a_vectors
        dst_index_by_id = self.b_index_by_id if a_to_b else self.a_index_by_id
        dst_neihgbour_vects = [dst_vectors[dst_index_by_id[id]] for id in dst_card.neighbours]

        # calculate summary distance between src_neihgbour_vects and dst_neihgbour_vects
        dist = 0
        for i in range(len(src_neihgbour_vects)):
            vs, vd = src_neihgbour_vects[i], dst_neihgbour_vects[i]
            d = np.array([vs[j] - vd[j] for j in range(len(vs))])
            dist += np.linalg.norm(d)
        return dist

    def _find_closest_n_vectors(self,
                                src_vector: Tuple[float, ...],
                                dst_vectors: List[Tuple[float, ...]],
                                max_items: int) -> List[int]:
        sa = SortedItems(max_items)
        for i in range(len(dst_vectors)):
            dv = dst_vectors[i]
            d = np.array([src_vector[j] - dv[j] for j in range(len(src_vector))])
            dist = np.linalg.norm(d)
            sa.update(dist, i)

        return [it[0] for it in sa.items]

    def _build_vectors(self):
        self.lang_a_vectors = [self._build_word_vector(w) for w in self.lang_a_cards]
        self.lang_b_vectors = [self._build_word_vector(w) for w in self.lang_b_cards]
        # normalize vectors: let each coordinate varies in [0..1]
        self._normalize_vectors(self.lang_a_vectors)
        self._normalize_vectors(self.lang_b_vectors)

    @classmethod
    def _build_word_vector(cls, card: WordCard) -> Tuple[float, ...]:
        return card.vector_length, card.vector_variance, card.frequency, card.frequency_rel_rank, card.non_uniformity

    @classmethod
    def _normalize_vectors(cls, lang_vectors: List[Tuple[float, ...]]):
        first_v = lang_vectors[0]
        # find min and max for each column
        min_max = [(v, v, 0) for v in first_v]
        for v in lang_vectors:
            for i in range(len(v)):
                c = v[i]
                mi, ma, _ = min_max[i]
                min_max[i] = min(mi, c), max(ma, c), 0
        min_max = [(mi, ma, ma - mi if ma != mi else 1) for mi, ma, _r in min_max]

        # scale each value
        for idx in range(len(lang_vectors)):
            v = lang_vectors[idx]
            v = [(v[i] - min_max[i][0]) / min_max[i][2] for i in range(len(min_max))]
            lang_vectors[idx] = v
