from dataclasses import dataclass
import datetime
from typing import List, Optional
import codecs
import os
import json

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class OptimizationRecord:
    record_date: datetime.datetime = None
    score: float = None
    coeffs: List[float] = None

    def __str__(self):
        cfx = ','.join([f'{c}' for c in self.coeffs])
        return f'[{self.number}: {self.record_date}], score={self.score:.4f}, coeffs={cfx}'

    def __repr__(self):
        return str(self)


class OptimizationTrack:
    def __init__(self):
        self.default_path = os.path.dirname(os.path.realpath(__file__))
        self.default_path = os.path.join(self.default_path, '..', '..', 'data', 'optimization.json')
        self.records: Optional[List[OptimizationRecord]] = None

    def get_tracks(self) -> List[OptimizationRecord]:
        self._ensure_records()
        return self.records

    def add_record(self, record: OptimizationRecord):
        self._ensure_records()
        self.records.append(record)
        self._save_tracks()

    def _save_tracks(self):
        records = [r.to_dict() for r in self.records]
        with codecs.open(self.default_path, 'w', encoding='utf-8') as fw:
            fw.write(json.dumps(records))

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
        raw_records = json.loads(jsn)
        self.records = [OptimizationRecord.from_dict(r) for r in raw_records]


TRACK_SCORE = OptimizationTrack()
