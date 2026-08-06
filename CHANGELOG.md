# Changelog

## [0.3.0] - 2026-08-06

### Added
- Added support for 2D spatial metric visualization via `log_2d()`.
- Added 2D chart style toggle between scatter dots and line rendering.
- Added 1:1 aspect ratio toggle for 2D spatial metric charts.
- Added persistent run notes component in the dashboard UI with auto-saving to `notes.txt`.
- Added dynamic run color palette mapping for consistent run colors across selection changes.
- Added adaptive line styling and color tinting for grouped multi-metric charts.
- Added development workflow instructions to `README.md`.

### Changed
- Refactored record loading and chart option generation into generalized functions (`load_records`, `build_grouped_scalar_chart_options`, `build_2d_chart_options`).
- Refactored run selection state tracking to use native dictionary data binding.
- Updated dashboard UI cards, headers, and spacing layout.

### Fixed
- Fixed chart `xAxis` and `yAxis` title and label font colors to pure black for improved contrast.
- Fixed the github link

## [0.2.0] - 2026-08-05

### Added
- Added SVG chart rendering export support and configurable font family selection.
- Added `--reload` CLI flag for automatic reloading during local development.
- Added multi-column grid layout selector (1 to 4 columns) for responsive chart arrangements.

## [0.1.1] - 2026-08-05

### Changed
- Changed default browser page title to `xscope`.
- Removed default favicon icon from browser titlebar.
- Cleaned up startup console output to display only the direct dashboard URL.
