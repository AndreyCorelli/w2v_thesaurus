from typing import Any, Dict

from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

from corpus.dictionary_builder.constants import MAX_WORD_LEN


Base = declarative_base()


class WordCard(Base):
    __tablename__ = 'word_card'
    id = Column(Integer, primary_key=True)
    lang_code = Column(String(10))
    word = Column(Unicode(MAX_WORD_LEN))
    frequency = Column(Float)
    frequency_rank = Column(Integer)
    frequency_rel_rank = Column(Float)
    non_uniformity = Column(Float)
    vector_length = Column(Float)
    vector_variance = Column(Float)
    vector = Column(ARRAY(Float), unique=False)
    neighbours = Column(ARRAY(Integer), unique=False)
    UniqueConstraint('lang_code', 'word', name='lang_word')

    def __str__(self):
        return f'[{self.word}], f={self.frequency:.3f}, rank={self.frequency_rank}, u={self.non_uniformity:.2f}, ' + \
               f'Vl={self.vector_length:.2f}, Vv={self.vector_variance:.2f}'

    def __repr__(self):
        return str(self)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__
        if '_sa_instance_state' in d:
            del d['_sa_instance_state']
        return d
