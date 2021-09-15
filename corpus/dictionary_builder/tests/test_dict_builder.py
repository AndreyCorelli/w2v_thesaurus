import os
from unittest import TestCase

from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from corpus.dictionary_builder.corpus_reader import CorpusReader
from corpus.dictionary_builder.dictionary_builder import DictionaryBuilder
from corpus.dictionary_builder.lang_dictionary import LangDictionary

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '..', '..', '..', 'data', 'texts')


class TestDictBuilder(TestCase):
    def debug_word_neighbours_en(self):
        cards = CorpusFileManager().load('en').words
        w_0 = cards[0].word
        card_by_id = {c.id: c for c in cards}
        card_by_name = {c.word: c for c in cards}
        w_dog = card_by_name['dog']
        w_0_nb = [card_by_id[n] for n in cards[0].neighbours]
        w_dog_nb = [card_by_id[n] for n in w_dog.neighbours]
        self.assertGreater(len(w_0_nb), 1)

    def debug_word_neighbours_ru(self):
        cards = CorpusFileManager().load('ru').words
        w_0 = cards[0].word
        card_by_id = {c.id: c for c in cards}
        card_by_name = {c.word: c for c in cards}
        w_dog = card_by_name['бог']
        w_dog_vect = w_dog.vector
        w_0_nb = [card_by_id[n] for n in cards[0].neighbours]
        w_dog_nb = [card_by_id[n] for n in w_dog.neighbours]
        w_dog_nb_vect = w_dog_nb[0].vector
        self.assertGreater(len(w_0_nb), 1)

    def test_build_word_list(self):
        lang_code = 'ru'

        reader = CorpusReader()
        reader.read(os.path.join(DATA_PATH, lang_code), lang_code)
        builder = DictionaryBuilder()
        builder.build(reader.sentences, lang_code)
        cards = list(builder.card_by_word.values())
        sorted_cards = list(cards)
        sorted_cards.sort(key=lambda c: c.frequency_rank)
        assert len(cards) > 10
        CorpusFileManager().save(LangDictionary(lang_code, builder.cards))
