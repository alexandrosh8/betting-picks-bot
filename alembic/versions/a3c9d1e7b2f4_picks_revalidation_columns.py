"""picks revalidation columns — current odds/edge refreshed each poll

Revision ID: a3c9d1e7b2f4
Revises: bc9e18be0148
Create Date: 2026-06-11
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a3c9d1e7b2f4"
down_revision: str | Sequence[str] | None = "bc9e18be0148"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("picks", sa.Column("current_odds", sa.Numeric(10, 4), nullable=True))
    op.add_column("picks", sa.Column("current_edge", sa.Numeric(12, 6), nullable=True))
    op.add_column("picks", sa.Column("revalidated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("picks", "revalidated_at")
    op.drop_column("picks", "current_edge")
    op.drop_column("picks", "current_odds")
