from typing import List, Dict, Tuple, Optional
import numpy as np

from corpus.dictionary_builder.sorted_items import SortedItems
from corpus.models import WordCard
from translator.metaparams import METAPARAMS
from translator.synonym_finder_params import SynonymFinderParams


class SynonymsFound:
    def __init__(self):
        self.synonyms: List[str] = []
        self.message = ''

    def __repr__(self):
        return self.message if self.message else ', '.join([f'{s}' for s in self.synonyms])

    def __str__(self):
        return self.__repr__()


class SynonymFinder:
    DEFAULT_SEARCH_PARAMS = SynonymFinderParams()

    def __init__(self,
                 lang_a_cards: List[WordCard],
                 lang_b_cards: List[WordCard]):
        self.lang_a_cards = lang_a_cards
        self.lang_b_cards = lang_b_cards

        # { word[i].word: i, ... }
        self.a_index_by_word = {lang_a_cards[i].word: i for i in range(len(lang_a_cards))}
        self.b_index_by_word = {lang_b_cards[i].word: i for i in range(len(lang_b_cards))}

        # { word: word_card, ... }
        self.card_a_by_word: Dict[str, WordCard] = {w.word: w for w in lang_a_cards}
        self.card_b_by_word: Dict[str, WordCard] = {w.word: w for w in lang_b_cards}

        # [word[0].vector, ..., word[N-1].vector]
        self.lang_a_vectors: List[Tuple[float, ...]] = []
        self.lang_b_vectors: List[Tuple[float, ...]] = []

        # [word[0].is_stem, ... ]
        self.a_stem_flags: List[bool] = [c.is_stem() for c in lang_a_cards]
        self.b_stem_flags: List[bool] = [c.is_stem() for c in lang_b_cards]

        self._build_world_2_vector_maps()

    def _get_word_stem(self, a_to_b: bool, wrd: str) -> Optional[str]:
        words = self.card_a_by_word if a_to_b else self.card_b_by_word
        card = words.get(wrd)
        return card.word if card else None

    def find_synonyms(self,
                      a_to_b: bool = True,
                      wrd: str = '',
                      search_params: SynonymFinderParams = None) -> SynonymsFound:
        search_params = search_params or self.DEFAULT_SEARCH_PARAMS

        result = SynonymsFound()
        if search_params.find_for_word_stem:
            wrd = self._get_word_stem(a_to_b, wrd)
            if not wrd:
                result.message = 'Source word not found'
                return result

        src_idx = self.a_index_by_word.get(wrd) if a_to_b else self.b_index_by_word.get(wrd)
        if src_idx is None:
            result.message = 'Source word not found'
            return result

        src_card = self.lang_a_cards[src_idx] if a_to_b else self.lang_b_cards[src_idx]
        src_vectors = self.lang_a_vectors if a_to_b else self.lang_b_vectors
        dst_vectors = self.lang_b_vectors if a_to_b else self.lang_a_vectors
        stem_flags = self.b_stem_flags if a_to_b else self.a_stem_flags

        # find <synonym_count> closest vectors in dst_vectors
        candidate_ids = self._find_n_closest_words_by_vectors(
            src_vectors[src_idx], dst_vectors, search_params.synonym_count,
            True, stem_flags)
        # get neighbours of the src item
        src_index_by_word = self.a_index_by_word if a_to_b else self.b_index_by_word
        src_neihgbour_vects = [src_vectors[src_index_by_word[w]] for w in src_card.neighbours]

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
        dst_index_by_word = self.b_index_by_word if a_to_b else self.a_index_by_word
        dst_neihgbour_vects = [dst_vectors[dst_index_by_word[w]] for w in dst_card.neighbours]

        # calculate summary distance between src_neihgbour_vects and dst_neihgbour_vects
        dist = 0
        for i in range(len(src_neihgbour_vects)):
            vs, vd = src_neihgbour_vects[i], dst_neihgbour_vects[i]
            d = np.array([vs[j] - vd[j] for j in range(len(vs))])
            dist += np.linalg.norm(d)
        return dist

    def _find_n_closest_words_by_vectors(self,
                                         src_vector: Tuple[float, ...],
                                         dst_vectors: List[Tuple[float, ...]],
                                         max_items: int,
                                         check_stems_only: bool = False,
                                         stem_flags: Optional[List[bool]] = None) -> List[int]:
        penalty_distance = 10**6
        sa = SortedItems(max_items)
        for i in range(len(dst_vectors)):
            if check_stems_only and not stem_flags[i]:
                dist = penalty_distance
            else:
                dv = dst_vectors[i]
                d = np.array([src_vector[j] - dv[j] for j in range(len(src_vector))])
                dist = np.linalg.norm(d)
            sa.update(dist, i)

        return [it[0] for it in sa.items]

    def _build_world_2_vector_maps(self):
        self.lang_a_vectors = [self._build_word_vector(w) for w in self.lang_a_cards]
        self.lang_b_vectors = [self._build_word_vector(w) for w in self.lang_b_cards]
        # normalize vectors: let each coordinate varies in [0..1]
        self._normalize_vectors(self.lang_a_vectors)
        self._normalize_vectors(self.lang_b_vectors)

    @classmethod
    def _build_word_vector(cls, card: WordCard) -> Tuple[float, ...]:
        v = (card.vector_length, card.vector_variance, card.frequency,
             card.non_uniformity,
             card.rel_length, card.prob_repeats)
        return tuple(v[i] * METAPARAMS.word_vector_weights[i] for i in range(len(v)))

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
