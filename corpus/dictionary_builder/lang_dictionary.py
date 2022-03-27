from typing import List, Dict, Optional, Tuple

from corpus.models import WordCard


class LangDictionary:
    def __init__(self,
                 lang_code: str,
                 words: Optional[List[WordCard]] = None,
                 word_to_stem: Optional[Dict[str, str]] = None):
        self.words: List[WordCard] = words or []
        self.lang_code = lang_code
        self.word_to_stem: Dict[str, str] = word_to_stem or {}

    def __str__(self):
        return f'{self.lang_code}: {len(self.words)} words'

    def __repr__(self):
        return str(self)

    @classmethod
    def unwrap_stem_words(cls, stem_words: List[Tuple[str, List[str]]]) -> Dict[str, str]:
        word_to_stem = {}
        for stem, words in stem_words:
            for word in words:
                word_to_stem[word] = stem
        return word_to_stem

    def wrap_stem_words(self) -> List[Tuple[str, List[str]]]:
        stems = {}
        for word, stem in self.word_to_stem.items():
            wrd_list = stems.get(stem)
            if not wrd_list:
                stems[stem] = [word]
            else:
                wrd_list.append(word)

        return [(stm, wrds) for stm, wrds in stems.items()]
