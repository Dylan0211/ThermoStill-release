# ThermoStill

Implementation of **ThermoStill: Distilling Time Series Foundation Model into Thermal Dynamics Model for HVAC Model Predictive Control**.

This repository contains the ThermoStill training pipeline for HVAC thermal dynamics modeling, including RC student models, teacher wrappers, RL-based teacher weighting, training scripts, and result logging.

## Overview

ThermoStill distills predictions from multiple time-series foundation models into a physics-informed RC thermal dynamics model. This repository focuses on the thermal modeling stage used in the paper.

The codebase supports:

- RC student models: `R1C1`, `R2C1`, `R2C2`
- teacher models: `chronos`, `timesfm`, `timemoe`
- end-to-end ThermoStill training and evaluation
- structured logs, metric export, prediction plots, and cached teacher predictions

## Requirements

- Python 3.10+
- PyTorch-compatible environment
- ecobee building data prepared into `house_data_csvs0/`
- solar covariates from NREL NSRDB
- the Python packages listed in [requirements.txt](/D:/Dropbox/liangrui-personal/26_CodeFiles/ThermoStill/requirements.txt)

Install the environment with:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data Preparation

### 1. Download the ecobee dataset

The ecobee dataset is public, but it is not distributed with this repository.

Download it from:

- [ecobee dataset](https://bbd.labworks.org/ds/bbd/ecobee)

### 2. Download solar data

The ecobee dataset does not include solar covariates. Solar data should be downloaded separately from NREL NSRDB:

- [NREL NSRDB](https://developer.nrel.gov/docs/solar/nsrdb/)

### 3. Build `house_data_csvs0`

The directory `house_data_csvs0/` is an intermediate processed dataset rather than the raw ecobee release. It should be constructed using the covariate-preparation logic from:

- [sample_ecobee_for_covariate.py](https://github.com/nesl/TSFM_Building/blob/main/sample_ecobee_for_covariate.py)

After this step, your local directory should contain ecobee house CSV files under `data/raw/ecobee/house_data_csvs0/`.

### 4. Build `house_data_by_state`

This repository includes a lightweight preprocessing step that groups the processed house files by state and converts them into the format used by ThermoStill.

Run:

```powershell
cd data\raw\ecobee
python preprocess.py
```

This creates:

```text
data/raw/ecobee/house_data_by_state/<STATE>/house_id_....csv
```

Each training CSV is expected to contain these columns:

- `time`
- `T01_TEMP`
- `Text`
- `duty_cycle`
- `GHI`

## Quick Start

Minimal example:

```powershell
python main.py `
  --state_dataset TX `
  --file_name house_id_da09897f6b67c4511ee33c658ddbdfe3afd082e3.csv
```

Example with explicit model settings:

```powershell
python main.py `
  --state_dataset TX `
  --file_name house_id_da09897f6b67c4511ee33c658ddbdfe3afd082e3.csv `
  --rc_model R1C1 `
  --tsfm_name_list chronos timesfm timemoe `
  --pretrain_epochs 20 `
  --max_epochs 200
```

## Shell Script

A shell entrypoint is provided in [run_thermostill.sh](/D:/Dropbox/liangrui-personal/26_CodeFiles/ThermoStill/scripts/run_thermostill.sh):

```bash
bash scripts/run_thermostill.sh
```

You can override its defaults through environment variables:

```bash
STATE_DATASET=CA \
RC_MODEL=R2C1 \
MAX_EPOCHS=50 \
DEVICE=cuda:0 \
bash scripts/run_thermostill.sh
```

## Outputs

Running the training pipeline writes artifacts to:

- `logs/thermostill/`: structured training logs
- `results/thermostill/`: exported evaluation metrics in JSON format
- `graphs/thermostill/`: prediction plots
- `tmp_data/`: cached teacher predictions
- `checkpoints/thermostill/`: saved model checkpoints

## Repository Structure

- [main.py](/D:/Dropbox/liangrui-personal/26_CodeFiles/ThermoStill/main.py): command-line entrypoint
- [exp/exp_thermostill.py](/D:/Dropbox/liangrui-personal/26_CodeFiles/ThermoStill/exp/exp_thermostill.py): end-to-end training loop
- [models/grey_box](/D:/Dropbox/liangrui-personal/26_CodeFiles/ThermoStill/models/grey_box): RC thermal dynamics models
- [models/rl](/D:/Dropbox/liangrui-personal/26_CodeFiles/ThermoStill/models/rl): actor, critic, and temporal encoder
- [models/tsfm/tsfm.py](/D:/Dropbox/liangrui-personal/26_CodeFiles/ThermoStill/models/tsfm/tsfm.py): teacher wrappers
- [data_provider](/D:/Dropbox/liangrui-personal/26_CodeFiles/ThermoStill/data_provider): data loading and cached teacher predictions
- [data/raw/ecobee/preprocess.py](/D:/Dropbox/liangrui-personal/26_CodeFiles/ThermoStill/data/raw/ecobee/preprocess.py): project-side ecobee preprocessing

## Citation

If you use this repository in academic work, please cite the ThermoStill paper.