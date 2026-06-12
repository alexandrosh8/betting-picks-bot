"""picks anchor_type column — fair-value anchor per pick

'pinnacle' | 'sharp' (named non-Pinnacle sharp book) | 'consensus'
(>=3-book median fallback, app/edge/value.py::anchor_type_for). Lets live
CLV be stratified by the anchor that produced each pick — the live verdict
mechanism for the consensus fallback (train-only validation 2026-06-12,
.claude/memory/decisions.md). Nullable VARCHAR(16): NULL = model-strategy
pick or a row persisted before this column existed (never backfilled — the
anchor of a historical pick is recoverable only from reason_summary).

Revision ID: e9b3c7a1d5f2
Revises: d7e2f5a8c1b9
Create Date: 2026-06-12
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e9b3c7a1d5f2"
down_revision: str | Sequence[str] | None = "d7e2f5a8c1b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "picks",
        sa.Column("anchor_type", sa.String(16), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("picks", "anchor_type")
