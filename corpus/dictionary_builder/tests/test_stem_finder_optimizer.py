from unittest import TestCase

from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.optimization.root_finder.stem_finder_optimizer import StemFinderOptimizer, StemFinderOptimizerParams


class TestStemFinderOptimizer(TestCase):
    def test_optimize(self):
        lang = 'en'
        cards = CorpusFileManager().load(lang).words
        ld = LangDictionary(lang, cards)

        opt = StemFinderOptimizer(ld)
        opt.optimize(StemFinderOptimizerParams())
