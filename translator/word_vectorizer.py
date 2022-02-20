from typing import Tuple, List, Dict

from corpus.dictionary_builder.alphabet import alphabet_by_code
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.models import WordCard
from translator.metaparams import METAPARAMS


class WordVectorizer:
    # the char is "rare" if it occurs less than
    # (text len in characters) / (alphabet len) * THRESHOLD_CHAR_SHARE
    THRESHOLD_CHAR_SHARE = 0.05

    def __init__(self):
        self.rare_characters: List[str] = []

    def vectorize_words(
            self, l_dict: LangDictionary) -> List[Tuple[float, ...]]:
        self._determine_rare_characters(l_dict)
        vectors = [self._build_word_vector(w) for w in l_dict.words]
        self._normalize_vectors(vectors)
        return vectors

    def _build_word_vector(self, card: WordCard) -> Tuple[float, ...]:
        has_rare_chars = 0
        for c in self.rare_characters:
            if c in card.word:
                has_rare_chars = 1
                break

        v = (card.vector_length,
             card.vector_variance,
             card.frequency,
             card.non_uniformity,
             card.rel_length,
             card.prob_repeats,
             has_rare_chars)
        return tuple(v[i] * METAPARAMS.word_vector_weights[i] for i in range(len(v)))

    def _determine_rare_characters(self, l_dict: LangDictionary):
        # calculate count of each character
        total_chars = 0
        char_count: Dict[str, int] = {}
        for w in l_dict.words:
            total_chars += len(w.word)
            for c in w.word:
                char_count[c] = char_count.get(c, 0) + 1

        # define rare characters
        chars_list = [(a, c) for a, c in char_count.items()]
        chars_list.sort(key=lambda cl: cl[1])

        threshold = total_chars / len(chars_list) * self.THRESHOLD_CHAR_SHARE
        self.rare_characters.append(chars_list[0][0])
        if chars_list[1][1] < threshold:
            self.rare_characters.append(chars_list[1][0])

    @classmethod
    def _normalize_vectors(cls, lang_vectors: List[Tuple[float, ...]]):
        first_v = lang_vectors[0]
        # find min and max for each column
        min_max = [(v, v, 0) for v in first_v]
        for v in lang_vectors:
            for i in range(len(v)):
                c = v[i]
                mi, ma, _ = min_max[i]
                min_max[i] = min(mi, c), max(ma, c), 0
        min_max = [(mi, ma, ma - mi if ma != mi else 1) for mi, ma, _r in min_max]

        # scale each value
        for idx in range(len(lang_vectors)):
            v = lang_vectors[idx]
            v = [(v[i] - min_max[i][0]) / min_max[i][2] for i in range(len(min_max))]
            lang_vectors[idx] = v
