from typing import Optional, List, Dict

from corpus.dictionary_builder.alphabet import Alphabet
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.dictionary_builder.vector_math import get_cosine_distance
from corpus.dictionary_builder.word_stemming.base_dictionary_word_stem_finder import BaseDictionaryWordStemFinder
from corpus.dictionary_builder.word_stemming.ngram_collector import MarginNgramsCollector
from corpus.models import WordCard


class StatisticsDictionaryWordStemFinder(BaseDictionaryWordStemFinder):
    """
    finds the words' "stems" using the word itself and the
    words vectors to sort out the "stems" that change the
    words' contexts too much
    """

    def __init__(self,
                 alphabet: Alphabet,
                 dictionary: LangDictionary):
        super().__init__(alphabet, dictionary)
        self.ngrams_collector = MarginNgramsCollector(alphabet, dictionary)
        self.ngrams_collector.build()
        self.all_words = {w.word for w in self.dictionary.words}
        self.word_2_root_distance_quantile = 0.4

    def find_stems(self):
        for word in self.dictionary.words:  # type: WordCard
            possible_roots = self._get_possible_stems(word.word)
            stem_word = self._check_possible_stems(possible_roots)
            if stem_word:
                word.stem = stem_word.stem
                # word.prefix = root_word.prefix
                # word.suffix = root_word.suffix
            else:
                word.stem = word.word
        self._filter_possible_roots()
        # self._count_roots()

    def _get_possible_stems(
            self,
            word: str) -> List[WordCard]:
        roots = {}  # Dict[str, WordCard]
        prefs = [None] + self.ngrams_collector.prefixes
        suffs = [None] + self.ngrams_collector.suffixes
        for pref in prefs:
            for suf in suffs:
                chopped = pref.chop_from_word(word, self.alphabet) if pref else word
                if not chopped:
                    continue
                chopped = suf.chop_from_word(chopped, self.alphabet) if suf else chopped
                if chopped:
                    card = WordCard()
                    card.word = word
                    card.stem = chopped
                    card.prefix = pref.text if pref else ''
                    card.suffix = suf.text if suf else ''
                    roots[f'{card.prefix}+{chopped}+{card.suffix}'] = card
        del roots[f'+{word}+']
        root_list = [roots[r] for r in roots]
        root_list.sort(key=lambda r: len(r.word))
        return root_list

    def _check_possible_stems(
            self,
            stems: List[WordCard]) -> Optional[WordCard]:
        prefixes = [None] + self.ngrams_collector.prefixes
        suffixes = [None] + self.ngrams_collector.suffixes

        for stem in stems:
            for pref in prefixes:
                for suf in suffixes:
                    modf = pref.text + stem.stem if pref else stem.stem
                    modf = modf + suf.text if suf else modf
                    if modf in self.all_words:
                        return stem
        return None

    def _count_roots(self):
        word_by_stem = {}  # type: Dict[str, List[WordCard]]
        for word in self.dictionary.words:
            if word.stem not in word_by_stem:
                word_by_stem[word.stem] = [word]
            else:
                word_by_stem[word.stem].append(word)
        for root in word_by_stem:
            root_count = sum([wr.count for wr in word_by_stem[root]])
            for word in word_by_stem[root]:
                word.root_count = root_count

    def _filter_possible_roots(self):
        """
        check the words' "roots": does the word vector remain more
        or less the same for the root?
        """
        if self.word_2_root_distance_quantile >= 1:
            return
        card_by_word = {w.word: w for w in self.dictionary.words}
        word2root_distances = {}
        missing_root_distance = 10 ** 5

        for w in self.dictionary.words:
            if w.stem == w.word:
                continue
            # calculate distance word-to-root
            root = card_by_word.get(w.stem)
            if not root:
                word2root_distances[w.word] = missing_root_distance
                continue
            w2r_distance = get_cosine_distance(w.vector, root.vector)
            word2root_distances[w.word] = w2r_distance

        # delete "roots" for the words with the w2r_distance
        # above the specified threshold
        distances = [d for _, d in word2root_distances.items() if d < missing_root_distance]
        distances.sort()
        max_distance = distances[int(len(distances) * self.word_2_root_distance_quantile)]

        for w in self.dictionary.words:
            w2r_d = word2root_distances.get(w.word)
            if not w2r_d or w2r_d < max_distance:
                continue
            # the word "root" is not a root because it changes the word's
            # "meaning" (or the context) significantly
            w.stem = w.word
