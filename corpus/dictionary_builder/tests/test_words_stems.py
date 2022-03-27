import os
from unittest import TestCase

from corpus.dictionary_builder.alphabet import alphabet_by_code
from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.dictionary_builder.word_stemming.dictionary_word_stem_finder import StatisticsDictionaryWordStemFinder
from corpus.dictionary_builder.word_stemming.nltk_dictionary_word_stem_finder import NltkDictionaryWordStemFinder


class TestWordsStems(TestCase):
    def test_debug_word_stemming(self):
        lang = 'en'
        cards = CorpusFileManager().load(lang).words
        alphabet = alphabet_by_code[lang]
        ld = LangDictionary(lang, cards)

        root_finder = StatisticsDictionaryWordStemFinder(alphabet, ld)
        root_finder.find_stems_in_dictionary()
        w_0 = cards[0].word
        CorpusFileManager().save(ld)

    def test_debug_word_stemming_nltk(self):
        lang = 'en'
        mgr = CorpusFileManager()

        cards = mgr.load(lang).words
        alphabet = alphabet_by_code[lang]
        ld = LangDictionary(lang, cards)

        root_finder = NltkDictionaryWordStemFinder(alphabet, ld)
        root_finder.find_stems_in_dictionary()
        w_0 = cards[0].word

        save_path = mgr.get_file_path("", ld.lang_code)
        save_folder = os.path.dirname(save_path) + "_nltk"
        mgr.save(ld, save_folder)
