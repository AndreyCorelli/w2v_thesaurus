from typing import List, Dict, Optional

from corpus.models import WordCard


class LangDictionary:
    def __init__(self,
                 lang_code: str,
                 words: Optional[List[WordCard]] = None):
        self.words: List[WordCard] = words or []
        self.lang_code = lang_code

    def __str__(self):
        return f'{self.lang_code}: {len(self.words)} words'

    def __repr__(self):
        return str(self)
