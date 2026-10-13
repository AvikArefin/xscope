# Changelog

## [0.6.0] - 2026-10-14

### Fixed
- The diagnostics panel no longer scrolls sideways: NiceGUI's footer style laid its header and results out in a row.
- Epoch 0 is plotted at x = 0; it was drawn at x = 1, on top of epoch 1.
- Text values in `metrics.jsonl` (e.g. `{"phase": "warmup"}`) no longer become broken charts.
- A corrupt line in a `.jsonl` file is skipped; before, nothing after it was ever loaded.
- A new confusion-matrix step updates its chart in place; before, every chart on the page was rebuilt.
- "Select All" and "Clear All" draw the charts once instead of once per run.

### Changed
- "Run ended while still improving" now uses the run status and a slope fit over the final 10% of epochs, so
  cosine/step-decay schedules aren't flagged.
- Rewrote automated diagnostics as the `xscope.diagnostics` package (33 rules, see `DIAGNOSTICS_ANALYSIS.md`):
  - Metric roles are inferred from names (`train_loss`, `val/acc`, `valLoss`, `lr`, `grad_norm`, …) instead of hardcoded keys.
  - All thresholds are relative to each curve's own noise and improvement.
  - Rules no longer contradict each other (e.g. "overfitting" alongside "underfitting").
  - Healthy runs no longer raise warnings for convergence, dropout-style val < train, or saturated accuracy.
- Confusion matrices now always render with a square plot area. They scale with the number of classes, truncate long
  labels, rotate x labels, show counts only up to 15 classes (tooltip shows count and row %), use readable per-cell
  text colors, and render large matrices in a single pass.

### Added
- Run groups (`Tracker(group="v3")` in mtrick): the group is part of the run's label (`v3/100 #5`) and the run list
  gets a heading per group.
- Text logged with mtrick's `log_text` (`text.jsonl`) is shown below the charts, one table per run, newest first.
  Each table is a fixed height and scrolls inside itself, holds the newest 1000 records, and refreshes live
  without losing its scroll position.
- Diagnostics that use the new mtrick data (`config.json`, `status.json`, per-record `_time`, `calibration.jsonl`):
  crashed, interrupted or unresponsive runs; training slowdowns; memory creep; calibration (ECE, over/under-confidence);
  problems that line up with a config setting; config values stored with mixed types; outlier runs annotated with their
  config differences.
- More diagnostics from existing data: top runs within noise of each other; large train/val accuracy gap (memorization);
  logged accuracy vs confusion-matrix mismatch; per-class precision regression; 2D error drift along a sequence,
  single-axis error and scale bias; warmup and LR-decay hints on spikes and plateaus.
- New diagnostics:
  - **Scalar curves:** loss spikes (with learning-rate correlation), growing instability, not learning, plateau,
    exploding/vanishing gradients, runs stopped while still improving, epoch counter going backwards, frozen metrics,
    out-of-range metrics, impossible negative losses, vanished metrics.
  - **Confusion matrices:** trivial-baseline accuracy, accuracy hiding minority-class failure, over-predicted classes,
    mutual class confusion, per-class recall regression, changing eval-set size.
  - **2D:** outliers, over-dispersion, systematic offset, error regression.
  - **Cross-run:** metric-name mismatch, eval-set mismatch, uncommitted code.
- Diagnostics test suite (`uv run pytest`), including a false-alarm check on randomized healthy runs.

### Removed
- `FileState.add_record` and its record cap, which nothing called.
- Rules that misfired on healthy runs: absolute-slope "underfitting", accuracy "training stall", and the
  `epoch_progress` rule, which depended on a `target_epochs` field that mtrick never writes.

## [0.5.1] - 2026-09-16

### Changed
- Added Cortex Technologies presentation and studio attribution to the README.

## [0.5.0] - 2026-09-16

### Added
- Added automated diagnostics for training anomalies and model performance issues.
- Added Y-axis focus mode for chart inspection.

### Fixed
- Improved loss divergence, overfitting, underfitting, class confusion, and learning-rate diagnostics.
- Fixed 2D prediction divergence calculations to account for both axes.

## [0.4.0] - 2026-08-07
### Added
- efficient auto chart refresh

## [0.3.0] - 2026-08-06

### Added 
- confusion matrix support

### Fixed
- correction in README
- correction in CHANGELOG

## [0.2.0] - 2026-08-06

### Added
- Added support for 2D spatial metric visualization via `log_2d()`.
- Added 2D chart style toggle between scatter dots and line rendering.
- Added 1:1 aspect ratio toggle for 2D spatial metric charts.
- Added persistent run notes component in the dashboard UI with auto-saving to `notes.txt`.
- Added dynamic run color palette mapping for consistent run colors across selection changes.
- Added adaptive line styling and color tinting for grouped multi-metric charts.
- Added development workflow instructions to `README.md`.
- Added SVG chart rendering export support and configurable font family selection.
- Added `--reload` CLI flag for automatic reloading during local development.
- Added multi-column grid layout selector (1 to 4 columns) for responsive chart arrangements.


### Changed
- Refactored record loading and chart option generation into generalized functions (`load_records`, `build_grouped_scalar_chart_options`, `build_2d_chart_options`).
- Refactored run selection state tracking to use native dictionary data binding.
- Updated dashboard UI cards, headers, and spacing layout.

### Fixed
- Fixed chart `xAxis` and `yAxis` title and label font colors to pure black for improved contrast.
- Fixed the github link

## [0.1.1] - 2026-08-05

### Changed
- Changed default browser page title to `xscope`.
- Removed default favicon icon from browser titlebar.
- Cleaned up startup console output to display only the direct dashboard URL.
