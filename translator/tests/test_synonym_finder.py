from unittest import TestCase

from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from translator.synonym_finder import SynonymFinder
from translator.synonym_finder_params import SynonymFinderParams


class TestDictBuilder(TestCase):
    def test_word_neighbours_syn(self):
        mgr = CorpusFileManager()
        en_dict = mgr.load('en')
        ru_dict = mgr.load('ru')

        search_params = SynonymFinderParams()
        search_params.synonym_count = 3000
        search_params.depth = 1

        sf = SynonymFinder(en_dict, ru_dict)
        words = [
            ('god', True), ('i', True), ('profit', True),
            ('собака', False), ('мокрый', False),
            ('иду', False), ('мне', False)

        ]
        for wrd, dirct in words:
            snms = sf.find_synonyms(dirct, wrd)
            print(f'{wrd}: [{snms}]')
