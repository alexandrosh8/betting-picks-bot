"""picks closing_anchor_type — provenance of the CLOSE anchor (honest CLV)

``Pick.anchor_type`` records the CREATION anchor. The CLOSE, computed later by
the CLV true-up, can be anchored by a DIFFERENT book — a soft-book consensus
median when no named sharp book prices the market at close — yet that
consensus (or poll-time revalidation fallback) close was reported under the
creation anchor and counted in the headline stake-weighted CLV with no
provenance filter. This nullable varchar(16) records the anchor that produced
each close (pinnacle / sharp / consensus); together with ``closing_odds``
(NON-NULL = snapshot-sourced) it lets the per-anchor and headline CLV trust
only genuine sharp closes. Additive + nullable — rows closed before this
column stay NULL (feature-detected by the read path until populated).

Revision ID: d3b8f1c7a9e2
Revises: c2a7e4f1b8d6
Create Date: 2026-06-20
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d3b8f1c7a9e2"
down_revision: str | Sequence[str] | None = "c2a7e4f1b8d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "picks",
        sa.Column("closing_anchor_type", sa.String(length=16), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("picks", "closing_anchor_type")
