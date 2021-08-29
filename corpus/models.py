from sqlalchemy import *
from sqlalchemy.orm import (scoped_session, sessionmaker, relationship,
                            backref)
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///database.sqlite3', convert_unicode=True)
db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))

Base = declarative_base()
# We will need this for querying
Base.query = db_session.query_property()


class WordCard(Base):
    __tablename__ = 'word_card'
    id = Column(Integer, primary_key=True)
    lang_code = Column(String)
    word = Column(String)
    frequency = Column(Float)
    frequency_rank = Column(Integer)
    frequency_rel_rank = Column(Float)
    non_uniformity = Column(Float)
    vector_length = Column(Float)
    vector_variance = Column(Float)
    vector = Column(ARRAY(Float), unique=False)

    def __str__(self):
        return f'[{self.word}], f={self.frequency:.3f}, rank={self.frequency_rank}, u={self.non_uniformity:.2f}, ' + \
               f'Vl={self.vector_length:.2f}, Vv={self.vector_variance:.2f}'

    def __repr__(self):
        return str(self)
