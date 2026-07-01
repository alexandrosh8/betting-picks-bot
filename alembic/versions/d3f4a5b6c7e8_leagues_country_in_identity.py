"""leagues: country in identity (fix "Ethiopia - Premier League" country-merge)

Same-named leagues in DIFFERENT countries were merged under the first-seen
country because league identity was (sport_id, key) only — e.g. league_id=6
'Premier League' froze at country 'Ethiopia' and mislabeled ~67 Lebanese/
Kuwaiti/Mongolian/... fixtures on the picks feed (a DISPLAY/label defect; teams,
odds, and bet fields render correctly — no wrong-game or CLV corruption).

This makes COUNTRY part of league identity. Backfill NULL country -> '' (a NULL
is treated as DISTINCT by a Postgres unique index and would defeat dedup for
country-less sources), make country NOT NULL DEFAULT '', then swap the unique
constraint (sport_id, key) -> (sport_id, key, country).

NON-DESTRUCTIVE to the already-merged historical rows: they keep their frozen
country + league_id (no re-split). This only prevents FUTURE mis-merges — a new
"Premier League" fixture from a different country now creates its own league row.
The downgrade reverses the DDL (may require manual dedup if country-distinct
rows were added under the new constraint).

Revision ID: d3f4a5b6c7e8
Revises: b2c3d4e5f6a7
Create Date: 2026-07-01
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d3f4a5b6c7e8"
down_revision: str | Sequence[str] | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1) NULL country -> '' so the NOT NULL alter succeeds and the new unique index
    #    dedupes country-less rows (a NULL would read as distinct).
    op.execute("UPDATE leagues SET country = '' WHERE country IS NULL")
    op.alter_column(
        "leagues",
        "country",
        existing_type=sa.String(64),
        nullable=False,
        server_default="",
    )
    # 2) Swap the identity constraint. Existing rows are unique on (sport_id, key),
    #    so they are trivially unique on (sport_id, key, country) — no violation.
    op.drop_constraint("uq_leagues_sport_key", "leagues", type_="unique")
    op.create_unique_constraint(
        "uq_leagues_sport_key_country", "leagues", ["sport_id", "key", "country"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_leagues_sport_key_country", "leagues", type_="unique")
    op.create_unique_constraint("uq_leagues_sport_key", "leagues", ["sport_id", "key"])
    op.alter_column(
        "leagues",
        "country",
        existing_type=sa.String(64),
        nullable=True,
        server_default=None,
    )
