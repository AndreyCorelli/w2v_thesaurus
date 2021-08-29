from unittest import TestCase

from corpus.dictionary_builder.corpus_reader import CorpusReader
from corpus.dictionary_builder.dictionary_builder import DictionaryBuilder


class TestDictBuilder(TestCase):
    def test_build_word_list(self):
        reader = CorpusReader()
        reader.read('/home/andrey/sources/andrey/voynich_morph/vman/corpus/raw/en', 'en')

        builder = DictionaryBuilder()
        builder.build(reader.sentences, 'en')
        cards = list(builder.card_by_word.values())
        sorted_cards = list(cards)
        sorted_cards.sort(key=lambda c: c.frequency_rank)
        assert len(cards) > 10