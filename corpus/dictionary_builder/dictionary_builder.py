import math
from typing import List, Dict

from corpus.dictionary_builder.word_freq_card import WordFrequencyCard
from corpus.models import WordCard
from gensim.models import Word2Vec


class DictionaryBuilder:
    def __init__(self):
        self.sentences: List[List[str]] = []
        self.words: List[str] = []
        self.card_by_word: Dict[str, WordCard] = {}

    def build(self,
              sentences: List[List[str]],
              lang_code: str):
        self.sentences = sentences
        self.words = [item for sublist in sentences for item in sublist]
        unique_words = set(self.words)
        self.card_by_word = {w: WordCard(word=w, lang_code=lang_code) for w in unique_words}
        self._calculate_frequency_uniformity()
        self._calculate_vectors()

    def _calculate_frequency_uniformity(self):
        f_cards = WordFrequencyCard.calc_stat(self.words)
        for wrd in self.card_by_word:
            card = self.card_by_word[wrd]
            f_card = f_cards[wrd]
            card.frequency = f_card.frequency
            card.frequency_rank = f_card.rank
            card.frequency_rel_rank = f_card.rank / len(self.card_by_word)
            card.non_uniformity = f_card.nu

    def _calculate_vectors(self):
        model = Word2Vec(sentences=self.sentences, vector_size=100, window=5, min_count=1, workers=4)
        a = 1
        for wrd in self.card_by_word:
            card = self.card_by_word[wrd]
            v = model.wv[card.word]
            # calculate length and "mean deviation" for a vector
            mean_v = sum([i for i in v]) / len(v)
            sum_l, sum_dev = 0, 0
            for c in v:
                sum_l += c * c
                sum_dev += (c - mean_v) ** 2

            card.vector = v
            card.vector_length = math.sqrt(sum_l)
            card.vector_variance = math.sqrt(sum_dev / len(v)) / mean_v

