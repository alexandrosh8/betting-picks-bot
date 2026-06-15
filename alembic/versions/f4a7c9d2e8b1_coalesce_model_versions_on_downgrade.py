"""coalesce model_versions before old-key downgrade

Revision ID: f4a7c9d2e8b1
Revises: b2f6c4e9a1d3
Create Date: 2026-06-16
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f4a7c9d2e8b1"
down_revision: str | Sequence[str] | None = "b2f6c4e9a1d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Schema is already correct at b2f6c4e9a1d3. This revision exists so a
    # downgrade can prepare per-sport rows for the older global unique key.
    return None


def downgrade() -> None:
    # The older schema can store only one row per (name, version). Merge every
    # duplicate sport-specific row into the lowest id before the previous
    # migration recreates uq_model_versions_name_version.
    duplicate_rows = """
        SELECT
            id,
            MIN(id) OVER (PARTITION BY name, version) AS keep_id
        FROM model_versions
    """
    for table in ("model_predictions", "picks", "backtest_runs"):
        op.execute(
            sa.text(
                f"""
                WITH duplicate_model_versions AS (
                    SELECT id, keep_id
                    FROM ({duplicate_rows}) ranked
                    WHERE id <> keep_id
                )
                UPDATE {table}
                SET model_version_id = duplicate_model_versions.keep_id
                FROM duplicate_model_versions
                WHERE {table}.model_version_id = duplicate_model_versions.id
                """
            )
        )
    op.execute(
        sa.text(
            f"""
            WITH duplicate_model_versions AS (
                SELECT id
                FROM ({duplicate_rows}) ranked
                WHERE id <> keep_id
            )
            DELETE FROM model_versions
            USING duplicate_model_versions
            WHERE model_versions.id = duplicate_model_versions.id
            """
        )
    )
