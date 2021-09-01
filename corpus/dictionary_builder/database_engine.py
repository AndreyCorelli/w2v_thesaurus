from typing import Union

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.mock import MockConnection

from corpus.settings import SQL_ALCHEMY_URL


def create_db_engine() -> Union[MockConnection, Engine]:
    return create_engine(SQL_ALCHEMY_URL)
