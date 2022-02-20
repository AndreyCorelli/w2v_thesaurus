from typing import Optional, Dict, Set
import codecs

from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from translator.synonym_finder import SynonymFinder
from translator.synonym_finder_params import SynonymFinderParams


class TranslationPrecisionCalculator:
    def __init__(self,
                 expected_translations_dict_path: str,
                 lang_a: Optional[str] = None,
                 lang_b: Optional[str] = None,
                 dict_a: Optional[LangDictionary] = None,
                 dict_b: Optional[LangDictionary] = None,
                 search_params: Optional[SynonymFinderParams] = None):
        mgr = CorpusFileManager()

        self.dict_a = dict_a
        if not dict_a:
            cards_a = mgr.load(lang_a).words
            self.dict_a = LangDictionary(lang_a, cards_a)
        self.dict_b = dict_b
        if not dict_b:
            cards_b = mgr.load(lang_b).words
            self.dict_b = LangDictionary(lang_b, cards_b)
        self.translations = self._read_expected_translations(expected_translations_dict_path)
        self.synonym_finder = SynonymFinder(self.dict_a, self.dict_b)
        self.search_params = search_params or SynonymFinderParams()

    def calculate_precision(self,
                            synonyms_to_check: int = 5) -> float:
        hits, misses = 0, 0
        for word, translations in self.translations.items():
            sf = self.synonym_finder.find_synonyms(True, word, self.search_params)

            hit = False
            for s in sf.synonyms[:synonyms_to_check]:
                if s in translations:
                    hit = True
                    break
            if hit:
                hits += 1
            else:
                misses += 1
        return hits / (hits + misses)

    @classmethod
    def _read_expected_translations(cls, path: str) -> Dict[str, Set[str]]:
        translations: Dict[str, Set[str]] = {}
        with codecs.open(path, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                words = [w.lower().strip() for w in line.split(',') if w and w.strip()]
                if len(words) < 2:
                    continue
                translations[words[0]] = set(words[1:])
        return translations
