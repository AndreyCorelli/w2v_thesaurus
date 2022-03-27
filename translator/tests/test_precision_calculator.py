import os
from unittest import TestCase

from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from translator.optimization.precision_calculator import TranslationPrecisionCalculator
from translator.synonym_finder_params import SynonymFinderParams


class TestPrecisionCalculator(TestCase):
    OWN_PATH = os.path.dirname(os.path.realpath(__file__))

    def test_calculate_ru_en(self):
        translation_file_path = os.path.join(self.OWN_PATH, '..', '..', 'data',
                                             'optimization', 'translation_ru_en.csv')
        search_params = SynonymFinderParams()
        search_params.synonym_count = 120  # 120 -> 0.0091, 2520->0.0
        search_params.depth = 1  # 2 -> 0.00917
        calc = TranslationPrecisionCalculator(
            translation_file_path,
            lang_a='ru',
            lang_b='en',
            search_params=search_params)
        p = calc.calculate_precision()
        print(f"Precision: {p}")  # 0.0333

    def test_calculate_ru_en_nltk(self):
        translation_file_path = os.path.join(self.OWN_PATH, '..', '..', 'data',
                                             'optimization', 'translation_ru_en.csv')
        search_params = SynonymFinderParams()

        mgr = CorpusFileManager()
        load_path = mgr.get_file_path("", "en")
        load_folder = os.path.dirname(load_path) + "_nltk"

        dict_en = mgr.load("en", load_folder)
        calc = TranslationPrecisionCalculator(
            translation_file_path,
            lang_a='ru',
            dict_b=dict_en,
            search_params=search_params)
        p = calc.calculate_precision()
        print(f"Precision: {p}")  # 0.0333
