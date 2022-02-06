from dataclasses import dataclass
from typing import List
import os
import codecs

from dataclasses_json import dataclass_json


MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(MODULE_PATH, '..', 'data', 'metaparams.txt')


@dataclass_json
@dataclass
class Metaparams:
    VECTOR_LEN = 6
    word_vector_weights: List[float] = None

    @classmethod
    def create_default(cls) -> 'Metaparams':
        ptrs = Metaparams()
        ptrs.word_vector_weights = [1.0 / cls.VECTOR_LEN] * cls.VECTOR_LEN
        return ptrs

    def save(self):
        with codecs.open(MODULE_PATH, 'w', encoding='utf-8') as fw:
            fw.write(self.to_json())

    @classmethod
    def load(cls) -> 'Metaparams':
        if not os.path.isfile(MODULE_PATH):
            return Metaparams.create_default()
        with codecs.open(MODULE_PATH, 'r', encoding='utf-8') as fr:
            jsn = fr.read()
        if not jsn:
            return Metaparams.create_default()
        return Metaparams.from_json(jsn)


METAPARAMS = Metaparams.load()
