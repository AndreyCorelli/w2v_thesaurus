from unittest import TestCase

from corpus.dictionary_builder.corpus_reader import CorpusReader
from corpus.dictionary_builder.corpus_repository import get_corpus_repository
from corpus.dictionary_builder.dictionary_builder import DictionaryBuilder
from corpus.dictionary_builder.synonym_finder import SynonymFinder


class TestDictBuilder(TestCase):
    def test_word_neighbours_syn(self):
        cards_en = get_corpus_repository().get_cards_by_lang('en')
        cards_ru = get_corpus_repository().get_cards_by_lang('ru')
        sf = SynonymFinder(cards_en, cards_ru)
        s_god = sf.find_synonyms(True, 'god')
        s_i = sf.find_synonyms(True, 'i')
        s_go = sf.find_synonyms(False, 'иду')
        s_me = sf.find_synonyms(False, 'мне')

    def test_word_neighbours_en(self):
        cards = get_corpus_repository().get_cards_by_lang('en')
        w_0 = cards[0].word
        card_by_id = {c.id: c for c in cards}
        card_by_name = {c.word: c for c in cards}
        w_dog = card_by_name['dog']
        w_0_nb = [card_by_id[n] for n in cards[0].neighbours]
        w_dog_nb = [card_by_id[n] for n in w_dog.neighbours]
        self.assertGreater(len(w_0_nb), 1)

    def test_word_neighbours_ru(self):
        cards = get_corpus_repository().get_cards_by_lang('ru')
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
        reader = CorpusReader()
        #reader.read('/home/andrey/Downloads/pdf/ru', 'ru')
        reader.read('/home/andrey/sources/andrey/voynich_morph/vman/corpus/raw/ru', 'ru')

        builder = DictionaryBuilder()
        builder.build(reader.sentences, 'ru')
        cards = list(builder.card_by_word.values())
        sorted_cards = list(cards)
        sorted_cards.sort(key=lambda c: c.frequency_rank)
        assert len(cards) > 10
