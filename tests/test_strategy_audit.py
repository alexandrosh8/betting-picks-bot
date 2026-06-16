"""Strategy audit report: evidence-weighted recommendations, no live betting."""

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reports" / "strategy_audit.py"
_spec = importlib.util.spec_from_file_location("strategy_audit", _SCRIPT)
assert _spec is not None and _spec.loader is not None
strategy_audit = importlib.util.module_from_spec(_spec)
sys.modules["strategy_audit"] = strategy_audit
_spec.loader.exec_module(strategy_audit)


def _report() -> dict:
    return {
        "config": {
            "fallback_thr": 0.03,
            "min_odds": 1.6,
            "devig": "differential_margin_weighting",
        },
        "held_out": {
            "premium_ref": {"n": 61, "roi": 0.21, "clv_max": 0.08},
            "volume_ref": {"n": 360, "roi": 0.047, "clv_max": 0.015},
            "composite": {"n": 204, "roi": 0.039, "clv_max": 0.021},
            "premium_bootstrap": {"inc_clv_max": {"point": 0.079}},
            "volume_bootstrap": {"inc_clv_max": {"point": 0.014}},
            "composite_bootstrap": {"inc_clv_max": {"point": 0.020}},
            "strict_gate_pass": True,
            "verdict": "POSITIVE selection skill",
        },
    }


def _config(**updates):  # type: ignore[no-untyped-def]
    base = dict(
        pick_strategy="value",
        value_devig="differential_margin_weighting",
        value_min_edge=0.03,
        value_volume_min_edge=0.015,
        value_min_odds=1.6,
        value_ml_filter=False,
        value_ml_manifest_filename="value_filter_manifest.json",
        value_ml_model_filename="value_filter_model.txt",
        value_ml_manifest_allow_shadow=False,
    )
    base.update(updates)
    return strategy_audit.AuditConfig(**base)


def test_strategy_audit_keeps_high_roi_premium_threshold() -> None:
    audit = strategy_audit.build_audit(
        _config(),
        _report(),
        {"verdict": "ADOPT", "operating_point": {"q": 0.725}},
        {"verdict": "CANDIDATE"},
    )

    text = strategy_audit.render_markdown(audit)
    assert "Keep VALUE_MIN_EDGE=0.030" in text
    assert "Do not promote per-cell/composite thresholds to premium yet" in text
    assert "trial VALUE_ML_FILTER=true" in text
    assert "Do not enforce the v2 value-filter manifest yet" in text


def test_strategy_audit_flags_drifted_live_config() -> None:
    audit = strategy_audit.build_audit(
        _config(value_min_edge=0.015, value_devig="shin"),
        _report(),
        None,
        None,
    )

    recs = "\n".join(audit["recommendations"])
    assert "reset VALUE_MIN_EDGE to 0.030" in recs
    assert "Align VALUE_DEVIG" in recs
