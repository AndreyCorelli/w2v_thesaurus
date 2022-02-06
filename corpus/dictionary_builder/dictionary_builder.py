import math
from typing import List, Dict, Tuple

from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.dictionary_builder.word_freq_card import WordFrequencyCard
from corpus.models import WordCard
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DictionaryBuilder:
    NEIGHBOURS_COUNT = 20

    def __init__(self):
        self.sentences: List[List[str]] = []
        self.words: List[str] = []
        self.cards: List[WordCard] = []
        self.card_by_word: Dict[str, WordCard] = {}

    def build(self,
              sentences: List[List[str]],
              lang_code: str) -> LangDictionary:
        self.sentences = sentences
        self.words = [item for sublist in sentences for item in sublist]
        unique_words = set(self.words)
        self.card_by_word = {w: WordCard(word=w, lang_code=lang_code) for w in unique_words}
        self.cards = list(self.card_by_word.values())
        self._calculate_frequency_uniformity()
        self._calculate_vectors()
        self._find_neighbours()
        return LangDictionary(lang_code, self.cards)

    def _calculate_frequency_uniformity(self):
        f_cards = WordFrequencyCard.calc_stat(self.words)
        for card in self.cards:
            f_card = f_cards[card.word]
            card.rel_length = f_card.rel_len
            card.prob_repeats = f_card.prob_repeats
            card.frequency = f_card.frequency
            card.frequency_rank = f_card.rank
            card.frequency_rel_rank = f_card.rank / len(self.card_by_word)
            card.non_uniformity = f_card.nu

    def _calculate_vectors(self):
        model = Word2Vec(sentences=self.sentences, vector_size=100, window=5, min_count=1, workers=4)
        for card in self.cards:
            v = model.wv[card.word]
            # calculate length and "mean deviation" for a vector
            mean_v = sum([i for i in v]) / len(v)
            sum_l, sum_dev = 0, 0
            for c in v:
                sum_l += c * c
                sum_dev += (c - mean_v) ** 2

            card.vector = v.astype(float)
            card.vector_length = math.sqrt(sum_l)
            card.vector_variance = math.sqrt(sum_dev / len(v))  # / mean_v

    def _find_neighbours(self):
        part_size = 15000
        for i in range(math.ceil(len(self.cards) / part_size)):
            self._find_neighbours_part(i, part_size)

    def _find_neighbours_part(self, start_index: int, size: int):
        start_card = start_index * size
        src_vectors = np.array([w.vector.astype(np.float32) for w in
                                self.cards[start_card: start_card + size]])
        all_neighbours: List[List[Tuple[int, float]]] = [[] for _ in range(size)]
        for index in range(start_index, start_index + 100000):
            start = index * size
            end = min(len(self.cards), (index+1)*size)
            dst_vectors = np.array([w.vector.astype(np.float32) for w in
                                    self.cards[start: end]])
            if len(dst_vectors) == 0:
                break
            similarities: np.ndarray = cosine_similarity(src_vectors, dst_vectors)

            for i in range(len(similarities)):
                shrunk = np.delete(similarities[i], i) if index == start_index else similarities[i]
                sort_indx = np.argsort(shrunk)
                index_sim = [(i + start, shrunk[i]) for i in sort_indx[-self.NEIGHBOURS_COUNT:]]
                all_neighbours[i] += index_sim

        for i in range(len(src_vectors)):
            all_neighbours[i].sort(key=lambda v: v[1], reverse=True)
            neib_indx = all_neighbours[i][:self.NEIGHBOURS_COUNT]
            self.cards[i + start_card].neighbours = [self.cards[ind].word for ind, _ in neib_indx]
