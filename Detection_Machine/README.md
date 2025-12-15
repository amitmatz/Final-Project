# Final Project – Detection Machine (LFP Speech Detection)

This repository contains the preprocessing and detection training pipeline for speech/non-speech detection from LFP signals.  
The pipeline supports multiple patients via a JSON config file and trains a Conv1D + BiLSTM + Attention model with 10-fold cross-validation.

## Project Components

- **Preprocessing**
  - Loads raw LFP `.mat` files per channel.
  - Exports per-channel CSV files (`time`, `signal`).
  - Loads label intervals from a tab-delimited labels file.
  - Aligns labels to absolute time using an offset extracted from `sound_w_times.mat`.
  - Saves `labels_aligned.csv` under the per-patient processed directory.
  - Optionally generates a legacy detection dataset `.npy` file for compatibility.

- **Detection Training**
  - Loads exported CSVs + `labels_aligned.csv`.
  - Extracts sliding windows over labeled regions and performs label cleanup using coverage thresholds.
  - Balances classes globally (as currently implemented).
  - Runs 10-fold cross-validation.
  - Selects the decision threshold on validation set (`accuracy` or `f1`).
  - Optional feature engineering (band-pass + smoothing).
  - Optional Focal Loss.
  - Optional post-hoc smoothing over prediction probabilities.

## Repository Layout (as used by the code)

Expected key files:

- `main.py`
- `Final_Project/Detection_machine/defines.py`
- `Final_Project/Detection_machine/preprocessing/preprocessing.py`
- `Final_Project/Detection_machine/Detection/train_detection.py`
- `config/patients_config.json`

Processed outputs (default):
- `processed_data/<Patient_ID>/csvs/*.csv`
- `processed_data/<Patient_ID>/labels_aligned.csv`
- `processed_data/<Patient_ID>_detection_data.npy` (legacy/compatibility output)

## Installation

1) Create and activate a virtual environment (recommended).
2) Install dependencies:

```bash
pip install -r requirements.txt
