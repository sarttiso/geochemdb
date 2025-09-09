# Changelog

## v1.1.0

### Added
- `GeochemDB.measurements_by_aliquot()`
- `GeochemDB.get_samples()`
- `GeochemDB.get_aliquots()`
- `GeochemDB.get_aliquots_samples()`

### Changed
- cursors are temporary
- check that Quantity-MeasurementUnit pairs exist before attempting to add measurements

## v1.0.0

- GeochemDB class for interacting with database
- aliquot_average() for averaging duplicate measurements