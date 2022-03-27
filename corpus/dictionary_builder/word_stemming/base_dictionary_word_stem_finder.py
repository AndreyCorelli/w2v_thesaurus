from typing import Optional


class BaseDictionaryWordStemFinder:
    """
    finds the words' "stems" using the word itself and the
    words vectors to sort out the "stems" that change the
    words' contexts too much
    """

    def __init__(self):
        ...

    def get_word_stem(self, word: str) -> Optional[str]:
        raise NotImplementedError()
