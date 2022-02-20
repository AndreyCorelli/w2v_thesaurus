from unittest import TestCase

from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from translator.synonym_finder import SynonymFinder
from translator.synonym_finder_params import SynonymFinderParams


class TestDictBuilder(TestCase):
    def test_word_neighbours_syn(self):
        mgr = CorpusFileManager()
        cards_en = mgr.load('en').words
        cards_ru = mgr.load('ru').words
        search_params = SynonymFinderParams()
        search_params.synonym_count = 100

        sf = SynonymFinder(
            LangDictionary('en', cards_en),
            LangDictionary('ru', cards_ru))
        s_god = sf.find_synonyms(True, 'god', search_params)
        s_i = sf.find_synonyms(True, 'i', search_params)
        s_go = sf.find_synonyms(False, 'иду', search_params)
        s_me = sf.find_synonyms(False, 'мне', search_params)
        s_none = sf.find_synonyms(False, 'несуществующееслово', search_params)
        assert not s_none.synonyms
        print(f"{s_god}\n{s_i}\n{s_go}\n{s_me}")
