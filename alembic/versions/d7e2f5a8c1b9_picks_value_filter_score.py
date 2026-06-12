"""picks value_filter_score column — meta-model annotation per pick

Calibrated value-filter meta-model score P(candidate beats the vig-free
Max-of-books close) from app/models/value_filter.py (meta-labeling SECONDARY
classifier; verdict ADOPT 2026-06-12, docs/research/ml-value-filter.md).
Nullable NUMERIC(8,6): NULL = artifact absent, ML deps not installed, or the
candidate is outside the model's trained scope (non-1x2/ou25 market, unmapped
league, consensus anchor, sub-1.6 odds). Historical rows stay NULL — scores
are never backfilled (the artifact that would score them did not exist when
they were picked).

Revision ID: d7e2f5a8c1b9
Revises: c1f4a9d27e3b
Create Date: 2026-06-12
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d7e2f5a8c1b9"
down_revision: str | Sequence[str] | None = "c1f4a9d27e3b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "picks",
        sa.Column("value_filter_score", sa.Numeric(8, 6), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("picks", "value_filter_score")
