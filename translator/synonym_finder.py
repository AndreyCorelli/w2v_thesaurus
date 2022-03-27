from typing import List, Dict, Tuple, Optional
import numpy as np

from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.dictionary_builder.sorted_items import SortedItems
from corpus.models import WordCard
from translator.synonym_finder_params import SynonymFinderParams
from translator.word_vectorizer import WordVectorizer


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
                 dict_a: LangDictionary,
                 dict_b: LangDictionary):
        lang_a_cards = dict_a.words
        lang_b_cards = dict_b.words

        self.dict_a = dict_a
        self.dict_b = dict_b
        self.lang_a_cards = dict_a.words
        self.lang_b_cards = dict_b.words

        # { word[i].word: i, ... }
        self.a_index_by_word = {lang_a_cards[i].word: i for i in range(len(lang_a_cards))}
        self.b_index_by_word = {lang_b_cards[i].word: i for i in range(len(lang_b_cards))}

        # { word: word_card, ... }
        self.card_a_by_word: Dict[str, WordCard] = {w.word: w for w in lang_a_cards}
        self.card_b_by_word: Dict[str, WordCard] = {w.word: w for w in lang_b_cards}

        # [word[0].vector, ..., word[N-1].vector]
        self.lang_a_vectors: List[Tuple[float, ...]] = []
        self.lang_b_vectors: List[Tuple[float, ...]] = []

        self._build_world_2_vector_maps()

    def find_synonyms(self,
                      a_to_b: bool = True,
                      wrd: str = '',
                      search_params: SynonymFinderParams = None) -> SynonymsFound:
        search_params = search_params or self.DEFAULT_SEARCH_PARAMS

        result = SynonymsFound()
        wrd = self._get_word_stem(a_to_b, wrd)
        if not wrd:
            result.message = 'Source word not found'
            return result

        src_idx = self.a_index_by_word.get(wrd) if a_to_b else self.b_index_by_word.get(wrd)
        if src_idx is None:
            result.message = 'Source word not found'
            return result

        src_vectors = self.lang_a_vectors if a_to_b else self.lang_b_vectors
        dst_vectors = self.lang_b_vectors if a_to_b else self.lang_a_vectors

        # find <synonym_count> closest vectors in dst_vectors
        candidate_ids = self._find_n_closest_words_by_vectors(
            src_vectors[src_idx], dst_vectors, search_params.synonym_count)

        candidate_dist: List[Tuple[int, float]] = []
        for candidate_id in candidate_ids:
            dst_nb_dist = self._get_distance_between_neighbours_iter(
                search_params.depth, src_idx, candidate_id, a_to_b,
                1, search_params)
            candidate_dist.append((candidate_id, dst_nb_dist))

        candidate_dist.sort(key=lambda cd: cd[1])
        dst_cards = self.lang_b_cards if a_to_b else self.lang_a_cards
        dst_words = [dst_cards[id].word for id, _ in candidate_dist]
        result.synonyms = dst_words
        return result

    def _get_word_stem(self, a_to_b: bool, wrd: str) -> Optional[str]:
        ld = self.dict_a if a_to_b else self.dict_b
        return ld.word_to_stem.get(wrd)

    def _get_distance_between_neighbours_iter(
            self,
            iterations_left: int,
            src_word_index: int,
            dst_word_index: int,
            a_to_b: bool,
            weight_multiplier: float,
            search_params: SynonymFinderParams = None) -> float:
        # TODO: neighbours count may not be the same
        src_card = self.lang_a_cards[src_word_index] if a_to_b else self.lang_b_cards[src_word_index]
        dst_card = self.lang_b_cards[dst_word_index] if a_to_b else self.lang_a_cards[dst_word_index]
        dst_vectors = self.lang_b_vectors if a_to_b else self.lang_a_vectors
        dst_index_by_word = self.b_index_by_word if a_to_b else self.a_index_by_word

        dst_neihgbour_indis = [dst_index_by_word[w] for w in dst_card.neighbours]
        dst_neihgbour_vects = [dst_vectors[ind] for ind in dst_neihgbour_indis]
        src_vectors = self.lang_a_vectors if a_to_b else self.lang_b_vectors
        src_index_by_word = self.a_index_by_word if a_to_b else self.b_index_by_word

        src_neihgbour_indis = [src_index_by_word[w] for w in src_card.neighbours]
        src_neihgbour_vects = [src_vectors[ind] for ind in src_neihgbour_indis]

        # calculate summary distance between src_neihgbour_vects and dst_neihgbour_vects
        dist = 0
        for i in range(len(src_neihgbour_vects)):
            # for each src word neighbour we calculate the distance
            # to the closest candidate's neighbour
            min_dist, min_index = -1, -1
            for j in range(len(dst_neihgbour_vects)):
                vs, vd = src_neihgbour_vects[i], dst_neihgbour_vects[j]
                d = self._get_vector_distance(vd, vs)

                if min_dist < 0 or min_dist > d:
                    min_dist = d
                    min_index = j
            dist += min_dist / len(src_neihgbour_vects) * weight_multiplier

            if iterations_left:
                dist += self._get_distance_between_neighbours_iter(
                    iterations_left - 1,
                    src_neihgbour_indis[i],
                    dst_neihgbour_indis[min_index],
                    a_to_b,
                    weight_multiplier * search_params.depth_weight_multiplier,
                    search_params
                )

        return dist

    @classmethod
    def _get_vector_distance(cls, vd, vs):
        d_v = np.array([vs[k] - vd[k] for k in range(len(vs))])
        d = np.linalg.norm(d_v)
        return d

    def _find_n_closest_words_by_vectors(
            self,
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

    def _build_world_2_vector_maps(self):
        vectorizer = WordVectorizer()
        self.lang_a_vectors = vectorizer.vectorize_words(self.dict_a)
        self.lang_b_vectors = vectorizer.vectorize_words(self.dict_b)
