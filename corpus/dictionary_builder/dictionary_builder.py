import collections
import math
from typing import List, Dict, Optional

from corpus.dictionary_builder.alphabet import alphabet_by_code
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.dictionary_builder.word_freq_card import WordFrequencyCard
from corpus.dictionary_builder.word_stemming.base_dictionary_word_stem_finder import BaseDictionaryWordStemFinder
from corpus.dictionary_builder.word_stemming.dictionary_word_stem_finder import StatisticsDictionaryWordStemFinder
from corpus.models import WordCard
from gensim.models import Word2Vec


class DictionaryBuilder:
    NEIGHBOURS_COUNT = 20

    def __init__(self,
                 stem_finder: Optional[BaseDictionaryWordStemFinder] = None):
        self.sentences: List[List[str]] = []
        self.words: List[str] = []
        self.cards: List[WordCard] = []
        self.card_by_word: Dict[str, WordCard] = {}
        self.stem_finder: Optional[BaseDictionaryWordStemFinder] = stem_finder
        self.lang_code = ''
        self.use_stems = False
        self.word_stems: Dict[str, str] = {}

    def build(self,
              sentences: List[List[str]],
              lang_code: str) -> LangDictionary:
        self.lang_code = lang_code
        self.sentences = sentences
        if self.stem_finder:
            self.sentences = self._get_word_stems_in_sentences(self.sentences)

        self._build_cards_with_vectors()

        if not self.stem_finder:
            # now we finally able to build the stem_finder
            # we stemmatize the words and them re-build dictionary
            self.stem_finder = StatisticsDictionaryWordStemFinder(
                alphabet_by_code[lang_code],
                LangDictionary(lang_code, self.cards)
            )
            self.sentences = self._get_word_stems_in_sentences(self.sentences)
            self._build_cards_with_vectors()

        self._find_words_neighbours()
        return LangDictionary(lang_code, self.cards, self.word_stems)

    def _get_word_stems_in_sentences(self, sentences: List[List[str]]) -> List[List[str]]:
        stem_sentences: List[List[str]] = []

        for s in sentences:
            new_sentence: List[str] = []
            for w in s:
                stem = self.word_stems.get(w)
                if not stem:
                    stem = self.stem_finder.get_word_stem(w)
                    self.word_stems[w] = stem
                new_sentence.append(stem)
            stem_sentences.append(new_sentence)
        return stem_sentences

    def _build_cards_with_vectors(self):
        words = [item for sublist in self.sentences for item in sublist]
        self.words = words
        unique_words = set(self.words)
        self.card_by_word = {w: WordCard(word=w) for w in unique_words}
        self.cards = list(self.card_by_word.values())
        self._calculate_frequency_uniformity()
        self._calculate_vectors()

    def _find_words_neighbours(self):
        # TODO: TF-IDF
        word_neighbours: Dict[str, Dict[str, float]] = {}
        window = collections.deque(maxlen=3)
        word_inv_frequency = {w.word: 1 / w.frequency**0.15 for w in self.cards}

        for w in self.words:
            w_if = word_inv_frequency[w]
            for win_word in window:
                if win_word == w:
                    continue
                word_nb = word_neighbours.get(win_word)
                if not word_nb:
                    word_neighbours[win_word] = {w: w_if}
                else:
                    counter = word_nb.get(w, 0)
                    word_nb[w] = counter + w_if
            window.append(w)
        for key_word in word_neighbours:
            card = self.card_by_word[key_word]
            all_words = list(word_neighbours[key_word].items())
            all_words.sort(key=lambda wc: wc[1], reverse=True)
            card.neighbours = [v for v, c in all_words[:self.NEIGHBOURS_COUNT]]

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
