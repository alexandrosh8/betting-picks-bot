"""picks tier column — two-tier picks (premium alerts vs volume CLV shadow)

tier='premium': edge >= VALUE_MIN_EDGE — full behavior (alert dispatch,
daily-exposure reservation). tier='volume': VALUE_VOLUME_MIN_EDGE <= edge <
premium — informational shadow tier persisted purely to accumulate live CLV
evidence (v2 holdout n=379, CLV +0.019); never alerted, never consumes the
exposure cap. Existing rows backfill to 'premium' via the server default —
every pre-tier pick was produced by the premium threshold.

Revision ID: c1f4a9d27e3b
Revises: f3a1c2d4e5b6
Create Date: 2026-06-11
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c1f4a9d27e3b"
down_revision: str | Sequence[str] | None = "f3a1c2d4e5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "picks",
        sa.Column("tier", sa.String(16), nullable=False, server_default="premium"),
    )


def downgrade() -> None:
    op.drop_column("picks", "tier")
