from dataclasses import dataclass
from typing import List
import os
import codecs

from dataclasses_json import dataclass_json


MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(MODULE_PATH, '..', '..', 'data', 'metaparams.txt')


@dataclass_json
@dataclass
class Metaparams:
    word_vector_weights: List[float] = None

    def __init__(self):
        self.word_vector_weights = [1.0 / 5] * 5

    def save(self):
        with codecs.open(MODULE_PATH, 'w', encoding='utf-8') as fw:
            fw.write(self.to_json())

    @classmethod
    def load(cls) -> 'Metaparams':
        if not os.path.isfile(MODULE_PATH):
            return Metaparams()
        with codecs.open(MODULE_PATH, 'w', encoding='utf-8') as fr:
            jsn = fr.read()
        if not jsn:
            return Metaparams()
        return Metaparams.from_json(jsn)


METAPARAMS = Metaparams.load()
