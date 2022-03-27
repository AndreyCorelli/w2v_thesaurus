from typing import Optional

from corpus.dictionary_builder.word_stemming.base_dictionary_word_stem_finder import BaseDictionaryWordStemFinder
from nltk.stem import PorterStemmer


class NltkDictionaryWordStemFinder(BaseDictionaryWordStemFinder):
    """
    finds the words' "stems" using NLTK
    """

    def __init__(self):
        super().__init__()
        self.ps = PorterStemmer()

    def get_word_stem(self, word: str) -> Optional[str]:
        return self.ps.stem(word)
