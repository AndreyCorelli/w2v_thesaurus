"""create word card

Revision ID: c825bf63f15c
Revises: 
Create Date: 2021-08-31 14:26:17.474986

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c825bf63f15c'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'word_card',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('lang_code', sa.String(10), nullable=False),
        sa.Column('word', sa.Unicode(50)),
        sa.Column('frequency', sa.Float),
        sa.Column('frequency_rank', sa.Integer),
        sa.Column('frequency_rel_rank', sa.Float),
        sa.Column('non_uniformity', sa.Float),
        sa.Column('vector_length', sa.Float),
        sa.Column('vector_variance', sa.Float),
        sa.Column('vector', sa.ARRAY(sa.Float)),
        sa.Column('neighbours', sa.ARRAY(sa.Integer)),
        sa.UniqueConstraint('lang_code', 'word', name='lang_word')
    )


def downgrade():
    op.drop_table('word_card')
