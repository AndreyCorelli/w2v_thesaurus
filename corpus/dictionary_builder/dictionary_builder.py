import math
from typing import List, Dict

from corpus.dictionary_builder.corpus_repository import get_corpus_repository
from corpus.dictionary_builder.word_freq_card import WordFrequencyCard
from corpus.models import WordCard
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DictionaryBuilder:
    NEIGHBOURS_COUNT = 10

    def __init__(self):
        self.sentences: List[List[str]] = []
        self.words: List[str] = []
        self.cards: List[WordCard] = []
        self.card_by_word: Dict[str, WordCard] = {}

    def build(self,
              sentences: List[List[str]],
              lang_code: str):
        self.sentences = sentences
        self.words = [item for sublist in sentences for item in sublist]
        unique_words = set(self.words)
        self.card_by_word = {w: WordCard(word=w, lang_code=lang_code) for w in unique_words}
        self.cards = list(self.card_by_word.values())
        self._calculate_frequency_uniformity()
        self._calculate_vectors()
        repo = get_corpus_repository()
        repo.clear_cards_by_lang(lang_code)
        repo.insert_cards(self.cards)
        self._find_neighbours()
        repo.update_neighbours(self.cards)

    def _calculate_frequency_uniformity(self):
        f_cards = WordFrequencyCard.calc_stat(self.words)
        for card in self.cards:
            f_card = f_cards[card.word]
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
            card.vector_variance = math.sqrt(sum_dev / len(v)) / mean_v

    def _find_neighbours(self):
        # this step helps with potential memory problem
        vectors = np.array([w.vector.astype(np.float32) for w in self.cards])
        # v_sparse = sparse.csr_matrix(vectors)
        similarities: np.ndarray = cosine_similarity(vectors)

        for i in range(len(similarities)):
            # shrunk = np.delete(similarities[i], i)
            shrunk = similarities[i]
            sort_indx = np.argsort(shrunk)
            neib_indx = sort_indx[-self.NEIGHBOURS_COUNT - 1: -1]
            self.cards[i].neighbours = [self.cards[ind].id for ind in neib_indx]

    @classmethod
    def _get_cosine_distance(cls, a: List[float], b: List[float]):
        ma, mb, ab = 0, 0, 0
        for i in range(len(a)):
            ab += a[i] * b[i]
            ma += a[i] * a[i]
            mb += b[i] * b[i]

        denom = math.sqrt(ma) * math.sqrt(mb)
        return ab / denom if denom > 0 else 0
