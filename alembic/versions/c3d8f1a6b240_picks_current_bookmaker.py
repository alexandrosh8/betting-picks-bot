"""picks current_bookmaker column — the book current_odds reflects

Live revalidation re-prices a pick at its OWN bookmaker by default, falling
back to the best remaining book only if the original book dropped the
selection (app/clv_trueup.py::revalidate_open_picks). This column records
which book the live current_odds came from, so the dashboard "now" line can
say "now at <book>" honestly in the rare fallback case instead of
mislabelling a different book's price as the pick's book. Nullable
VARCHAR(64): NULL = never revalidated, or a row persisted before this column.

Revision ID: c3d8f1a6b240
Revises: f4a7c9d2e8b1
Create Date: 2026-06-18
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c3d8f1a6b240"
down_revision: str | Sequence[str] | None = "f4a7c9d2e8b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "picks",
        sa.Column("current_bookmaker", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("picks", "current_bookmaker")
