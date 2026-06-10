# Modeling Notes

- Football MVP: Dixon-Coles (time-decay, low-score ρ correction) — ADR-0004
  once accepted; ML ensemble deferred to roadmap phase 6.
- NBA MVP: LightGBM-first with isotonic calibration — ADR-0005 once accepted.
- Calibration > accuracy for ROI: evaluate with Brier/log-loss + reliability
  diagrams, never raw accuracy.
- Devig defaults per market type: ADR-0006 once accepted.
- Leakage rules: features must use only information available at signal time;
  closing odds never appear in features.
