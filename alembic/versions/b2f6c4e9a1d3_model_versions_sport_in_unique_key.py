"""model_versions unique key gains sport_id

(name, version) -> (sport_id, name, version). The value strategy is sport-
agnostic and reuses one name/version ("value-sharp-vs-soft"/"v3") for soccer
AND basketball; the old key let the first sport's row win the sport_id and the
second sport silently reuse it (wrong sport attribution). Widening the key
(sport_id, name, version) is strictly safe: any set already unique on
(name, version) is unique on the superset, so no existing row can violate it.

Existing rows are left as-is — a model_versions row created during the shared
period keeps its (possibly mis-attributed) sport_id, and the picks already
pointing at it are not rewritten. Going forward each sport gets its own row.

Revision ID: b2f6c4e9a1d3
Revises: e9b3c7a1d5f2
Create Date: 2026-06-16
"""

from collections.abc import Sequence

from alembic import op

revision: str = "b2f6c4e9a1d3"
down_revision: str | Sequence[str] | None = "e9b3c7a1d5f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("uq_model_versions_name_version", "model_versions", type_="unique")
    op.create_unique_constraint(
        "uq_model_versions_sport_name_version",
        "model_versions",
        ["sport_id", "name", "version"],
    )


def downgrade() -> None:
    # Reversible only when no (name, version) is shared across sports — exactly
    # the state the upgrade was written to fix. If two sports share a strategy
    # row, the old narrow constraint cannot be recreated; resolve manually.
    op.drop_constraint("uq_model_versions_sport_name_version", "model_versions", type_="unique")
    op.create_unique_constraint(
        "uq_model_versions_name_version",
        "model_versions",
        ["name", "version"],
    )
