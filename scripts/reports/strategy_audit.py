"""Strategy audit report: live config vs backtest/control evidence.

Reads the existing offline artifacts under data/ml and prints an operator
report. It does not fetch data, train models, or change live thresholds.

Run:
    uv run python scripts/reports/strategy_audit.py

Decision-support only. Nothing here places bets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_REPORT = REPO_ROOT / "data" / "ml" / "threshold_control_report.json"
DEFAULT_VALUE_FILTER_MANIFEST = REPO_ROOT / "data" / "ml" / "value_filter_manifest.json"
DEFAULT_VALUE_FILTER_V2_MANIFEST = REPO_ROOT / "data" / "ml" / "value_filter_manifest_v2.json"


@dataclass(frozen=True)
class AuditConfig:
    pick_strategy: str
    value_devig: str
    value_min_edge: float
    value_volume_min_edge: float
    value_min_odds: float
    value_ml_filter: bool
    value_ml_manifest_filename: str
    value_ml_model_filename: str
    value_ml_manifest_allow_shadow: bool

    @classmethod
    def from_settings(cls) -> AuditConfig:
        settings = get_settings()
        return cls(
            pick_strategy=settings.pick_strategy,
            value_devig=settings.value_devig,
            value_min_edge=settings.value_min_edge,
            value_volume_min_edge=settings.value_volume_min_edge,
            value_min_odds=settings.value_min_odds,
            value_ml_filter=settings.value_ml_filter,
            value_ml_manifest_filename=settings.value_ml_manifest_filename,
            value_ml_model_filename=settings.value_ml_model_filename,
            value_ml_manifest_allow_shadow=settings.value_ml_manifest_allow_shadow,
        )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _held_row(report: dict[str, Any], key: str) -> dict[str, Any]:
    return dict(report.get("held_out", {}).get(key, {}))


def _inc_clv_max(report: dict[str, Any], key: str) -> float | None:
    row = report.get("held_out", {}).get(f"{key}_bootstrap", {}).get("inc_clv_max", {})
    point = row.get("point")
    return float(point) if isinstance(point, int | float) else None


def _recommendations(
    config: AuditConfig,
    threshold_report: dict[str, Any] | None,
    manifest_v1: dict[str, Any] | None,
    manifest_v2: dict[str, Any] | None,
) -> list[str]:
    recs: list[str] = []
    if config.pick_strategy != "value":
        recs.append(
            "Switch PICK_STRATEGY back to value for picks. The Dixon-Coles/model path "
            "is screens-only "
            "because prior backtests showed negative CLV."
        )

    if threshold_report is None:
        recs.append(
            "Rebuild the threshold-control artifact before changing strategy knobs: "
            "uv run python scripts/ml/optimize_thresholds.py"
        )
    else:
        held = threshold_report.get("held_out", {})
        premium = _held_row(threshold_report, "premium_ref")
        composite = _held_row(threshold_report, "composite")
        volume = _held_row(threshold_report, "volume_ref")
        premium_roi = float(premium.get("roi", 0.0))
        composite_roi = float(composite.get("roi", 0.0))
        volume_roi = float(volume.get("roi", 0.0))
        premium_inc = _inc_clv_max(threshold_report, "premium")
        composite_inc = _inc_clv_max(threshold_report, "composite")
        volume_inc = _inc_clv_max(threshold_report, "volume")
        fallback = float(threshold_report.get("config", {}).get("fallback_thr", 0.03))
        min_odds = float(threshold_report.get("config", {}).get("min_odds", 1.6))
        devig = str(threshold_report.get("config", {}).get("devig", ""))

        if config.value_min_edge != fallback:
            recs.append(
                f"For high-ROI premium alerts, reset VALUE_MIN_EDGE to {fallback:.3f}. "
                f"The premium reference ROI {_pct(premium_roi)} beat composite "
                f"{_pct(composite_roi)} "
                f"and volume {_pct(volume_roi)} on the held-out artifact."
            )
        else:
            recs.append(
                f"Keep VALUE_MIN_EDGE={config.value_min_edge:.3f} for premium alerts. "
                f"It is the highest-ROI reference in the current artifact "
                f"(premium ROI {_pct(premium_roi)}, incCLVmax {_num(premium_inc)})."
            )

        if config.value_volume_min_edge > fallback:
            recs.append(
                "Lower VALUE_VOLUME_MIN_EDGE back to 0.015 if you want shadow evidence. "
                "Volume rows do not alert, but they feed CLV learning."
            )
        if abs(config.value_min_odds - min_odds) > 1e-9:
            recs.append(
                f"Align VALUE_MIN_ODDS with the audited artifact ({min_odds:.2f}) "
                "before comparing ROI."
            )
        if config.value_devig != devig:
            recs.append(
                f"Align VALUE_DEVIG with the audited artifact ({devig}) before comparing ROI."
            )
        if composite_inc is not None and volume_inc is not None:
            recs.append(
                "Do not promote per-cell/composite thresholds to premium yet. "
                "They clear strict CLV as broader coverage "
                f"(composite incCLVmax {_num(composite_inc)}, "
                f"volume incCLVmax {_num(volume_inc)}) but they dilute ROI versus premium."
            )

        strict_gate = held.get("strict_gate_pass")
        if strict_gate is not True:
            recs.append(
                "Treat the threshold-control artifact as research only until strict "
                "incCLVmax passes."
            )

    verdict_v1 = str((manifest_v1 or {}).get("verdict", "")).upper()
    if verdict_v1 == "ADOPT":
        if config.value_ml_filter:
            recs.append(
                "VALUE_ML_FILTER is enabled with an ADOPT manifest. Keep monitoring "
                "live score-stratified "
                "CLV on the dashboard; disable if the high-score bucket stops beating close."
            )
        else:
            recs.append(
                "Next strictness lever: trial VALUE_ML_FILTER=true with the ADOPT v1 "
                "manifest for fewer, higher-quality alerts. Treat it as live "
                "evidence-gated: dashboard score buckets need "
                "sufficient n and positive CLV before making it permanent."
            )
    elif manifest_v1 is None:
        recs.append("No v1 value-filter manifest found; keep VALUE_ML_FILTER=false.")
    else:
        recs.append("Keep VALUE_ML_FILTER=false because the configured v1 manifest is not ADOPT.")

    verdict_v2 = str((manifest_v2 or {}).get("verdict", "")).upper()
    if verdict_v2 and verdict_v2 != "ADOPT":
        recs.append(
            "Do not enforce the v2 value-filter manifest yet. Its binding verdict is "
            "live shadow CLV + "
            "fresh 2627 season, not the spent 2425/2526 holdout."
        )
    return recs


def build_audit(
    config: AuditConfig,
    threshold_report: dict[str, Any] | None,
    manifest_v1: dict[str, Any] | None,
    manifest_v2: dict[str, Any] | None,
) -> dict[str, Any]:
    held = (threshold_report or {}).get("held_out", {})
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "live_config": config.__dict__,
        "backtest_refs": {
            "premium": _held_row(threshold_report or {}, "premium_ref"),
            "volume": _held_row(threshold_report or {}, "volume_ref"),
            "composite": _held_row(threshold_report or {}, "composite"),
            "strict_gate_pass": held.get("strict_gate_pass"),
            "verdict": held.get("verdict"),
        },
        "value_filter": {
            "v1_verdict": (manifest_v1 or {}).get("verdict"),
            "v1_operating_q": (manifest_v1 or {}).get("operating_point", {}).get("q"),
            "v2_verdict": (manifest_v2 or {}).get("verdict"),
        },
        "recommendations": _recommendations(config, threshold_report, manifest_v1, manifest_v2),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    cfg = audit["live_config"]
    refs = audit["backtest_refs"]
    vf = audit["value_filter"]
    lines = [
        "# Strategy Audit",
        "",
        f"- Generated: `{audit['generated_at']}`",
        f"- Live strategy: `{cfg['pick_strategy']}`",
        f"- Live premium edge: `{cfg['value_min_edge']:.3f}`",
        f"- Live volume edge: `{cfg['value_volume_min_edge']:.3f}`",
        f"- Live odds floor: `{cfg['value_min_odds']:.2f}`",
        f"- Live devig: `{cfg['value_devig']}`",
        f"- ML filter enforced: `{cfg['value_ml_filter']}`",
        "",
        "## Backtest References",
        "",
    ]
    for label in ("premium", "volume", "composite"):
        row = refs.get(label) or {}
        lines.append(
            f"- {label}: n `{row.get('n', 'n/a')}`, ROI `{_pct(row.get('roi'))}`, "
            f"CLVmax `{_num(row.get('clv_max'))}`"
        )
    lines.extend(
        [
            f"- Strict gate pass: `{refs.get('strict_gate_pass')}`",
            f"- Verdict: {refs.get('verdict') or 'n/a'}",
            "",
            "## Value Filter",
            "",
            f"- v1 verdict: `{vf.get('v1_verdict')}`; q*: `{vf.get('v1_operating_q')}`",
            f"- v2 verdict: `{vf.get('v2_verdict')}`",
            "",
            "## Recommendations",
            "",
        ]
    )
    lines.extend(f"{i}. {rec}" for i, rec in enumerate(audit["recommendations"], start=1))
    lines.extend(
        [
            "",
            "Manual review required. This system does not place bets, and ROI is not guaranteed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-report", type=Path, default=DEFAULT_THRESHOLD_REPORT)
    parser.add_argument("--manifest-v1", type=Path, default=DEFAULT_VALUE_FILTER_MANIFEST)
    parser.add_argument("--manifest-v2", type=Path, default=DEFAULT_VALUE_FILTER_V2_MANIFEST)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    parser.add_argument("--out", type=Path, default=None, help="optional output file")
    args = parser.parse_args()

    audit = build_audit(
        AuditConfig.from_settings(),
        _load_json(args.threshold_report),
        _load_json(args.manifest_v1),
        _load_json(args.manifest_v2),
    )
    text = json.dumps(audit, indent=2) + "\n" if args.json else render_markdown(audit)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
