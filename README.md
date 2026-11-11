# Crowd Behaviour Research Validation

Research workspace for evaluating a reconstructed RSKEF-inspired crowd anomaly detection approach.

## Project Structure

- `src/` - Validation runners and supporting scripts.
- `r_skef_core/` - Local implementation of motion features, GMC, calibration and configuration.
- `UMN/` - Dataset location (dataset files are not included).
- `Results/UMN/` - Candidate annotations and validation outputs.

## Current Experimental Status

- Dataset: UMN Crowd Activity
- Video: 7,739 frames at 30 FPS
- Candidate sequences: 11
- Configurations: C1-C6
- Calibration observations: 90 per sequence
- Smoke testing: Completed

## Important Research Limitation

The current UMN annotations are provisional candidates and have not yet been independently verified against authoritative frame-level ground truth.

Consequently, results based on these annotations must be treated as exploratory rather than validated detector performance.

## Reproducibility

The original dataset must be obtained separately and placed in the expected `UMN/` directory.

Review the experiment metadata and validation scripts before running experiments.
