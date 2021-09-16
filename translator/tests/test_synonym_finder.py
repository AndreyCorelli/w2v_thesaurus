from unittest import TestCase

from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from translator.synonym_finder import SynonymFinder


class TestDictBuilder(TestCase):
    def test_word_neighbours_syn(self):
        mgr = CorpusFileManager()
        cards_en = mgr.load('en').words
        cards_ru = mgr.load('ru').words
        sf = SynonymFinder(cards_en, cards_ru)
        s_god = sf.find_synonyms(True, 'god')
        s_i = sf.find_synonyms(True, 'i')
        s_go = sf.find_synonyms(False, 'иду')
        s_me = sf.find_synonyms(False, 'мне')