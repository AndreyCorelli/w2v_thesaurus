from typing import List

from sqlalchemy.orm import declarative_base, Session

from corpus.dictionary_builder.database_engine import create_db_engine
from corpus.models import WordCard


class CorpusRepository:
    def __init__(self):
        self._base = declarative_base()

    def get_cards_by_lang(self, lang_code: str) -> List[WordCard]:
        engine = create_db_engine()
        with Session(engine) as session:
            query = session.query(WordCard).filter(WordCard.lang_code == lang_code)
            return list(query)

    def clear_cards_by_lang(self, lang_code: str):
        engine = create_db_engine()
        with Session(engine) as session:
            session.query(WordCard).filter(WordCard.lang_code==lang_code).delete()
            session.commit()

    def insert_cards(self, cards: List[WordCard]):
        engine = create_db_engine()
        with Session(engine) as session:
            session.bulk_save_objects(cards, return_defaults=True)
            session.commit()

    def update_neighbours(self, cards: List[WordCard]):
        engine = create_db_engine()
        with Session(engine) as session:
            session.bulk_update_mappings(
                WordCard,
                [dict(id=c.id, neighbours=c.neighbours) for c in cards])
            session.commit()


DEFAULT_CORPUS_REPOSITORY = CorpusRepository()


def get_corpus_repository() -> CorpusRepository:
    return DEFAULT_CORPUS_REPOSITORY