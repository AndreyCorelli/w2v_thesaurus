from corpus.dictionary_builder.alphabet import Alphabet
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.dictionary_builder.word_stemming.base_dictionary_word_stem_finder import BaseDictionaryWordStemFinder
from nltk.stem import PorterStemmer


class NltkDictionaryWordStemFinder(BaseDictionaryWordStemFinder):
    """
    finds the words' "stems" using NLTK
    """

    def __init__(self,
                 alphabet: Alphabet,
                 dictionary: LangDictionary):
        super().__init__(alphabet, dictionary)

    def find_stems(self):
        ps = PorterStemmer()
        for word in self.dictionary.words:
            word.stem = ps.stem(word.word)
