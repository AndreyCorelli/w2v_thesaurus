from typing import List, Optional

from corpus.models import WordCard


class TranslatedWord:
    def __init__(self,
                 word_card: WordCard,
                 start: int,
                 end: int,
                 synonyms: Optional[List[str]] = None):
        self.word_card = word_card
        self.start = start
        self.end = end
        self.synonyms = synonyms

    def __str__(self):
        snm = ', '.join((self.synonyms or [])[:3]) + '..'
        return f'{self.word_card.word}, f_rel: {self.word_card.frequency_rel_rank}, [{snm}]'

    def __repr__(self):
        return str(self)


class Translation:
    def __init__(self,
                 src_path: str,
                 src_lang_code: str,
                 dst_lang_code: str):
        self.src_path = src_path
        self.src_lang_code = src_lang_code
        self.dst_lang_code = dst_lang_code
        self.translated: List[TranslatedWord] = []

