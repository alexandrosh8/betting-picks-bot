"""pick_line_drift — vig-free fair drift time-series per pick (build #6 / C8)

A new APPEND-ONLY table recording the de-vigged fair probability + the implied
CLV-so-far at each re-price of an open pick, from bet-time to close. The ``picks``
row keeps only a single close snapshot (closing_fair_probability / clv_log); this
preserves the whole drift PATH for good/bad-variance attribution + steam analysis.

ADDITIVE — creates one new table, touches no existing table. Rows are written ONLY
when ``CLV_RECORD_DRIFT`` is enabled (default OFF), so applying this migration is
inert: the table simply stays empty until the flag is turned on.

Revision ID: b4f2a9c83d1e
Revises: 2d37faf2d3fd
Create Date: 2026-06-27
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b4f2a9c83d1e"
down_revision: str | Sequence[str] | None = "2d37faf2d3fd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pick_line_drift",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("pick_id", sa.BigInteger(), sa.ForeignKey("picks.id"), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fair_probability", sa.Numeric(8, 6), nullable=False),
        sa.Column("fair_odds", sa.Numeric(10, 4), nullable=True),
        sa.Column("clv_log", sa.Numeric(12, 6), nullable=True),
        sa.Column("anchor_type", sa.String(length=16), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_pick_line_drift_pick", "pick_line_drift", ["pick_id", "captured_at"])


def downgrade() -> None:
    op.drop_index("idx_pick_line_drift_pick", table_name="pick_line_drift")
    op.drop_table("pick_line_drift")
