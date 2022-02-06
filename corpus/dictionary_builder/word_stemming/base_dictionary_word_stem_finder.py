from corpus.dictionary_builder.alphabet import Alphabet
from corpus.dictionary_builder.lang_dictionary import LangDictionary


class BaseDictionaryWordStemFinder:
    """
    finds the words' "stems" using the word itself and the
    words vectors to sort out the "stems" that change the
    words' contexts too much
    """

    def __init__(self,
                 alphabet: Alphabet,
                 dictionary: LangDictionary):
        self.alphabet = alphabet
        self.dictionary = dictionary

    def find_stems(self):
        raise NotImplementedError()
