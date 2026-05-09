# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — Round 2 fixes

### Added
- `src/risk_metrics.py` — module computing Sharpe, Sortino, Calmar,
  max drawdown, annualised volatility and total/annualised return
  from a portfolio-value trajectory. Numerically validated against
  closed-form known cases in `tests/test_metrics.py`.
- `src/significance_tests.py` — one-sided binomial test and
  permutation test for directional accuracy, with Wilson 95%
  confidence intervals.
- `tests/` — pytest suite with three modules:
  - `test_no_leakage.py`: chronological-split, train-only scaler,
    and look-ahead-bias regression checks.
  - `test_metrics.py`: 14 unit tests pinning the math of all
    risk metrics and significance tests.
  - `test_gatekeeper.py`: ticker whitelist, filename sanitisation,
    rate-limiter and SHA-256-is-not-a-defense regression checks.
- `evaluation_graph.png` — dual-panel test-set visualisation
  (price with Buy/Sell/Hold markers + portfolio trajectory). Now
  embedded in the README.

### Changed
- README.md rewritten in academic tone:
  - Emoji removed from all section headers.
  - "Production-Ready", "17.8x improvement", "All course standards
    met" and similar marketing phrases replaced with neutral
    technical language.
  - "Honest Assessment" / "Project Summary & Achievements" /
    "Files & Deliverables" promotional sections removed; replaced
    with one consolidated "Limitations" subsection.
  - Win-rate metric split into two clearly named quantities used
    consistently throughout: **Positive-Step Rate** (% of
    timesteps with positive portfolio change) and **Trade Win
    Rate** (% of profitable closed positions).
  - Directional accuracy now reported with a binomial test
    p-value and a Wilson 95% confidence interval.
  - Security section corrected — SHA-256 ticker hashing is
    described as a logging-privacy measure, not as an
    injection-prevention mechanism. Whitelist regex
    (`^[A-Z]{1,5}(\.[A-Z])?$`) is now stated as the actual
    injection defense.
  - Abstract reformatted to a single coherent paragraph.

### Fixed (carried over from earlier commits — for the record)
- **Data leakage**: Min-Max scaler is fit only on the training
  slice prior to applying to the test slice (commit `05df107`).
- **Risk metrics in evaluation**: `eval_results.csv` now stores
  Sharpe / MDD / Calmar computed from the actual test-set
  portfolio (commit `ff08d59`).
- **Gatekeeper ticker validation**: whitelist regex now allows
  optional letter suffix for tickers such as `BRK.B` (commit
  `e200c01`).

### Outstanding (deferred)
- Re-run end-to-end training and re-generate
  `evaluation_graph.png`, `eval_results.csv` and the headline
  metrics in the README from the post-leakage-fix model. The
  numbers currently in the README are inherited from the
  pre-fix run; until retraining completes, the directional
  accuracy figure must be interpreted as provisional.
