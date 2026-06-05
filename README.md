Python 3.14 or higher

pandas (2.0.3), numpy (1.24.3), scipy (1.10.1), matplotlib (3.7.1), openpyxl (3.1.2)

License
Apache License 2.0

Author
Tiakat

# Blood Pressure Analysis: Invasive vs Non-Invasive Monitoring in Surgery

This repository contains the complete Python codebase for the PROMISES study, a prospective observational study of 50 surgical patients comparing invasive arterial blood pressure monitoring (arterial catheter) with non-invasive oscillometric cuff measurements.

## Key Finding

After advanced artifact detection using the Signal Abnormality Index (SAI) algorithm and extensive data cleaning, patients spent an average of **33.6%** of surgery time with clinically significant differences (>10 mmHg) between the two measurement methods.

## Study Overview

- **Population**: 50 adult patients undergoing elective non-cardiac surgery
- **Data sources**: Drager Infinity monitoring system + NOL monitor event files
- **Artifact detection**: SAI algorithm (Sun et al. 2006) with ±10 beat expansion window
- **Primary outcome**: Percentage of intraoperative time with |MAP_invasive - MAP_non-invasive| ≥ 10 mmHg

## Scripts (in execution order)

| Script | Description |
|--------|-------------|
| `transformed-time-extracted-nbp.py` | Converts OBSERVATION_DATETIME from YYYYMMDDHHMMSS to HH:MM:SS format and extracts NBP columns |
| `sai_pipeline_without_artifacts.py` | Applies SAI algorithm, expands artifact window (±10 beats), generates 6-panel diagnostic PNGs |
| `sai-csv-cleaned.py` | Batch version - exports cleaned invasive CSV files without diagnostic figures |
| `onlyfor-noniva-extraction.py` | Extracts non-invasive data, keeps operation segment, removes * character from NBP values |
| `superposed_filtered.py` | Plots invasive (blue line) vs non-invasive (black circles), shades red where |difference| ≥ 10 mmHg |
| `repeat-operation-inv-noinv.py` | Shades only between two consecutive different NBP values where BOTH have |gradient| ≥ 10 mmHg |
| `gradient.py` | Calculates gradient statistics, exports to formatted Excel with conditional formatting |
| `temps-gradient.py` | Alternative calculation using only changed NBP values (excludes repeated identical readings) |
| `histogram.py` | Generates distribution plots of gradient percentages across all patients |
| `variable.py` | Performs univariable analysis with unit verification (range checks, cm→mm conversion) |
| `multivarie.py` | Performs multivariable linear regression with forest plot and predicted-vs-observed plot |

## Requirements

```bash
pip install pandas numpy scipy matplotlib openpyxl

