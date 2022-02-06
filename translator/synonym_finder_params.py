class SynonymFinderParams:
    def __init__(self,
                 synonym_count: int = 20,
                 find_for_word_stem: bool = False):
        self.synonym_count = synonym_count
        self.find_for_word_stem = find_for_word_stem

    def __repr__(self) -> str:
        return f"[synonym_count: {self.synonym_count}; "\
               f"find_for_word_stem:{self.find_for_word_stem}]"
