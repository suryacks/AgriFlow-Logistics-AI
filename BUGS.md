# Known Bugs / Issues

## [UI-001] Layout Readability & Spacing
**Status**: In Progress (Fixing in current sprint)
**Description**: The UI components, particularly the Alpha Charts and System Internals, appear squished vertically and lack sufficient horizontal spacing on some resolutions. The "layered" text in System Internals was unreadable.
**Fix Plan**:
- Enable global scrolling (Done).
- Increase horizontal gaps (gap-4 -> gap-8).
- Increase padding (p-4 -> p-6).
- Implement responsive grid sizing.

## [ALPH-002] Prediction Variance
**Status**: Resolved
**Description**: Profit estimates were identical across all dates.
**Fix**: Implemented date-seeded random generation for deterministic but unique daily forecasts.
