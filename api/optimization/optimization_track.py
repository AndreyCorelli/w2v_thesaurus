from dataclasses import dataclass
import datetime
from typing import List, Optional, Tuple, Dict
import codecs
import os
import json

from dataclasses_json import dataclass_json

from corpus.dictionary_builder.corpus_file_manager import CorpusFileManager
from corpus.dictionary_builder.lang_dictionary import LangDictionary
from corpus.models import WordCard
from translator.synonym_finder import SynonymFinder


class DatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def date_hook(json_dict):
    for (key, value) in json_dict.items():
        try:
            json_dict[key] = datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
        except:
            pass
    return json_dict


@dataclass_json
@dataclass
class OptimizationRecord:
    # TODO: Object of type datetime is not JSON serializable
    record_date: datetime.datetime = None
    score: float = None
    coeffs: List[float] = None

    def __str__(self):
        cfx = ','.join([f'{c}' for c in self.coeffs])
        return f'[{self.number}: {self.record_date}], score={self.score:.4f}, coeffs={cfx}'

    def __repr__(self):
        return str(self)

    @classmethod
    def json_converter(cls, o):
        if isinstance(o, datetime.datetime):
            return o.__str__()


@dataclass_json
@dataclass
class EvaluationSample:
    word: str = None
    translations: List[str] = None
    score: int = 0

    def __str__(self):
        return ', '.join([f'{w}: {tr}' for w, tr in self.translations[:3]]) + ' ...'

    def __repr__(self):
        return str(self)

    @property
    def translations_str(self) -> str:
        return ' '.join(self.translations)


@dataclass_json
@dataclass
class EvaluationSampleSet:
    words_by_lang: Dict[Tuple[str, str], List[EvaluationSample]] = None

    def __str__(self):
        return ', '.join([f'{k}: {len(self.words_by_lang[k])}'
                          for k in self.words_by_lang])

    def __repr__(self):
        return str(self)

    @property
    def score(self) -> Optional[float]:
        if not self.words_by_lang:
            return None
        score = 0.0
        for w_lst in self.words_by_lang.values():
            score += sum([wrd.score for wrd in w_lst])
        return score


class OptimizationTrack:
    def __init__(self):
        self.own_path = os.path.dirname(os.path.realpath(__file__))
        self.default_path = os.path.join(self.own_path, '..', '..', 'data', 'optimization', 'optimization.json')
        self.sample_path = os.path.join(self.own_path, '..', '..', 'data', 'optimization',
                                        '../../data/optimization/test_samples.txt')
        self.records: Optional[List[OptimizationRecord]] = None

    def get_tracks(self) -> List[OptimizationRecord]:
        self._ensure_records()
        return self.records

    def add_record(self, record: OptimizationRecord):
        self._ensure_records()
        self.records.append(record)
        self._save_tracks()

    def prepare_sample_set(self) -> EvaluationSampleSet:
        with open(self.sample_path) as f:
            raw = json.load(f)
        st = EvaluationSampleSet()
        st.words_by_lang = {}
        cards_by_lang: Dict[str, List[WordCard]] = {}
        finder_by_langs: Dict[Tuple[str, str], SynonymFinder] = {}
        mgr = CorpusFileManager()

        for record in raw:
            src_lang, dst_lang = record['src_lang'], record['dest_lang']
            key = (src_lang, dst_lang)
            a2b = True
            finder = finder_by_langs.get(key)
            if not finder:
                a2b = False
                finder = finder_by_langs.get((dst_lang, src_lang))
            if not finder:
                a2b = True
                cards_src = cards_by_lang.get(src_lang)
                if not cards_src:
                    cards_src = mgr.load(src_lang).words
                    cards_by_lang[src_lang] = cards_src
                    cards_src = cards_by_lang.get(src_lang)
                cards_dst = cards_by_lang.get(dst_lang)
                if not cards_dst:
                    cards_dst = mgr.load(dst_lang).words
                    cards_by_lang[dst_lang] = cards_dst
                finder = SynonymFinder(
                    LangDictionary(src_lang, cards_src),
                    LangDictionary(dst_lang, cards_dst))
                finder_by_langs[key] = finder

            samples = []
            for w in record['words'].split(' '):
                sample = EvaluationSample()
                sample.translations = finder.find_synonyms(a2b, w).synonyms
                sample.word = w
                samples.append(sample)
            st.words_by_lang[key] = samples
        return st

    def _save_tracks(self):
        records = [r.to_dict() for r in self.records]
        with codecs.open(self.default_path, 'w', encoding='utf-8') as fw:
            fw.write(json.dumps(records, cls=DatetimeEncoder))

    def _ensure_records(self):
        if self.records is not None:
            return
        if not os.path.isfile(self.default_path):
            self.records = []
            return
        with codecs.open(self.default_path, 'r', encoding='utf-8') as fr:
            jsn = fr.read()
        if not jsn:
            self.records = []
            return
        raw_records = json.loads(jsn)  # , object_hook=date_hook)
        self.records = [OptimizationRecord.from_dict(r) for r in raw_records]


TRACK_SCORE = OptimizationTrack()
