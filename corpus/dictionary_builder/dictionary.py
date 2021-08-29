from typing import List, Dict

from corpus.models import WordCard


class Dictionary:
    def __init__(self):
        self.words: List[WordCard] = []
        self.card_by_word: Dict[str, WordCard] = {}
