from typing import Dict, Optional, Any

from corpus.dictionary_builder.alphabet import alphabet_by_code
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from nltk.stem import PorterStemmer
import numpy as np

from corpus.dictionary_builder.word_stemming.dictionary_word_stem_finder import StatisticsDictionaryWordStemFinder


class StemFinderOptimizerParams:
    def __init__(self,
                 min_quantile: float = 0.05,
                 max_quantile: float = 0.8,
                 quantile_step: float = 0.05):
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.quantile_step = quantile_step


class StemFinderOptimizer:

    def __init__(self,
                 lang_dict: LangDictionary):
        self.lang_dict = lang_dict
        self.stem_by_word: Dict[str, str] = {}
        self.optimize_params = StemFinderOptimizerParams()
        self.orig_stems: Dict[str, str] = {}

    def optimize(self,
                 optimize_params=StemFinderOptimizerParams) -> Dict[str, Any]:
        self.optimize_params = optimize_params
        self._find_word_stems_nltk()
        alphabet = alphabet_by_code[self.lang_dict.lang_code]

        root_finder = StatisticsDictionaryWordStemFinder(alphabet, self.lang_dict, 1)
        root_finder.find_stems()
        self.orig_stems = {c.word: c.stem for c in self.lang_dict.words}
        return self._do_optimize(root_finder)

    def _do_optimize(self,
                     root_finder: StatisticsDictionaryWordStemFinder) -> Dict[str, Any]:
        best_params = {"quantile": None}
        best_precision: Optional[float] = None

        for quantile in np.arange(
                self.optimize_params.min_quantile,
                self.optimize_params.max_quantile,
                self.optimize_params.quantile_step):
            root_finder.word_2_root_distance_quantile = quantile
            root_finder.filter_possible_roots()
            precision = self._calculate_precision()
            if best_precision is None or precision > best_precision:
                best_precision = precision
                best_params["quantile"] = quantile

            for card in self.lang_dict.words:
                card.stem = self.orig_stems[card.word]

        print(f"Optimized params: {best_params}")
        return best_params


    def _calculate_precision(self) -> float:
        hit, miss = 0, 0
        for card in self.lang_dict.words:
            is_hit = card.stem == self.stem_by_word[card.word]
            if is_hit:
                hit += 1
            else:
                miss += 1
        return hit / (hit + miss)

    def _find_word_stems_nltk(self):
        ps = PorterStemmer()
        for card in self.lang_dict.words:
            stem = ps.stem(card.word) or card.word
            self.stem_by_word[card.word] = stem
