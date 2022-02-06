import os
from typing import List
from unittest import TestCase

from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from corpus.dictionary_builder.corpus_reader import CorpusReader
from corpus.dictionary_builder.dictionary_builder import DictionaryBuilder
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.dictionary_builder.vector_math import get_cosine_distance

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '..', '..', '..', 'data', 'texts')


class TestDictBuilder(TestCase):
    def test_word_math(self):
        # power + man -> toned
        # wine - spirit -> antiq
        # lord - noble -> slimy
        # work + fun -> progresses
        cards = CorpusFileManager().load('en').words
        card_by_word = {c.word: c for c in cards}

        card0 = card_by_word['work']
        card1 = card_by_word['fun']
        v = self.sum_vectors(card0.vector, card1.vector, 1)
        resulted = self.find_closest(v, cards)
        print(resulted.word)

    def sum_vectors(self, va, vb, k):
        return [a + k * vb[i] for i, a in enumerate(va)]

    def find_closest(self, v: List[float], cards):
        distances = []
        for i, c in enumerate(cards):
            w2r_distance = get_cosine_distance(c.vector, v)
            distances.append((i, w2r_distance))
        distances.sort(key=lambda t: t[1])
        return cards[distances[0][0]]


    def debug_word_neighbours_en(self):
        cards = CorpusFileManager().load('en').words
        w_0 = cards[0].word
        card_by_name = {c.word: c for c in cards}
        w_dog = card_by_name['dog']
        w_0_nb = [card_by_name[n] for n in cards[0].neighbours]
        w_dog_nb = [card_by_name[n] for n in w_dog.neighbours]
        self.assertGreater(len(w_0_nb), 1)

    def test_debug_word_neighbours_ru(self):
        cards = CorpusFileManager().load('ru').words
        w_0 = cards[0].word
        card_by_name = {c.word: c for c in cards}
        w_dog = card_by_name['бог']
        w_dog_vect = w_dog.vector
        w_0_nb = [card_by_name[n] for n in cards[0].neighbours]
        w_dog_nb = [card_by_name[n] for n in w_dog.neighbours]
        w_dog_nb_vect = w_dog_nb[0].vector
        self.assertGreater(len(w_0_nb), 1)

    def test_build_word_list(self):
        lang_code = 'en'

        reader = CorpusReader()
        reader.read(os.path.join(DATA_PATH, lang_code), lang_code)
        builder = DictionaryBuilder()
        builder.build(reader.sentences, lang_code)
        cards = list(builder.card_by_word.values())
        sorted_cards = list(cards)
        sorted_cards.sort(key=lambda c: c.frequency_rank)
        assert len(cards) > 10
        CorpusFileManager().save(LangDictionary(lang_code, builder.cards))
