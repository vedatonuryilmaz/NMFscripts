# NMF K-Optimization Pipeline for ATAC-seq Data

## Overview

Python-based CLI pipeline for Non-negative Matrix Factorization (NMF) k-optimization on ATAC-seq data. Processes TCGA Z-scores to identify optimal component numbers through:

1. Binary accessibility mapping from Z-scores
2. Sample grouping by embryonic origin
3. NMF evaluation across k-values (2-26)
4. Metric calculation (F1, AUPRC, Reconstruction Error)
5. Model weight storage and visualization

## Quick Start

```bash
# Setup
python3 -m venv nmf_env && source nmf_env/bin/activate
pip install -r requirements.txt

# Configure paths in config.json
# Run pipeline
python prepare_data.py
python run_group_nmf_cli.py Ectoderm
python run_allsamples_nmf_cli.py
```

## Repository Structure

```
├── config.json                    # Main configuration
├── emb.json                       # Sample groupings
├── requirements.txt               # Dependencies
├── data_utils.py                  # Data processing utilities
├── nmf_evaluation.py              # Evaluation metrics
├── nmf_plotting.py                # Visualization
├── nmf_workflow.py                # Core pipeline
├── prepare_data.py                # Data preparation CLI
├── run_group_nmf_cli.py           # Group-specific NMF CLI
├── run_allsamples_nmf_cli.py      # All-samples NMF CLI
└── embryonic_group_nmf_outputs_cli/
  ├── preprocessed_data/         # Intermediate files
  ├── Ect_NMF_K_opt_2_26/        # Ectoderm results
  ├── Mes_NMF_K_opt_2_26/        # Mesoderm results
  └── AllSamples_NMF_K_opt_2_26/ # Combined results
```

## Configuration

### `config.json` - Essential Settings
```json
{
  "TCGA_ZSCORES_PATH": "/path/to/TCGA_zscores.parquet",
  "K_RANGE_EMBRYONIC_START": 2,
  "K_RANGE_EMBRYONIC_END": 26,
  "TOP_N_FEATURES_CUTOFF": 117442,
  "N_JOBS_PARALLEL": -1
}
```

### `emb.json` - Sample Groupings
```json
{
  "organ_system_groupings": [
  {
    "group_name": "Ectoderm",
    "cancer_codes": ["BRCA", "SKCM", "HNSC"]
  }
  ]
}
```

## Input Requirements

- **TCGA Z-scores**: Parquet file with features as rows, samples as columns (starting column 7)
- **Sample IDs**: Must follow TCGA format for cancer type extraction
- **Python 3.7+** with packages from `requirements.txt`

## Usage

### Single Commands
```bash
# Data preparation (run once)
python prepare_data.py

# Group-specific analysis
python run_group_nmf_cli.py Ectoderm
python run_group_nmf_cli.py Mesoderm --n-jobs 4

# All samples analysis
python run_allsamples_nmf_cli.py
```

### Background Execution
```bash
# Using tmux
tmux new -s nmf_pipeline
source nmf_env/bin/activate
python prepare_data.py && \
python run_group_nmf_cli.py Ectoderm && \
python run_allsamples_nmf_cli.py
# Ctrl+b, d to detach
```

## Output Structure

Each analysis produces:
- **`{k}NMF/weights/`**: W.npy (features×k), H.npy (samples×k) matrices
- **`*_evaluation_summary.csv`**: Metrics for all k values
- **`summary_figures/`**: K-selection plots
- **`*_run_parameters.json`**: Analysis parameters

### Key Metrics
- **Max Mean F1**: Higher = better component separation
- **AUPRC**: Higher = better reconstruction quality  
- **Reconstruction Error**: Lower = better fit

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Wrong cancer type extraction | Modify `get_cancer_type_from_sample_id()` in `data_utils.py` |
| Memory issues | Reduce `N_JOBS_PARALLEL` or `BOOL_MAP_CHUNK_SIZE` |
| Path errors | Verify all paths in `config.json` |
| Polars issues | Update Polars version, check `enable_string_cache()` usage |

## FAIR Compliance

- **Findable**: Clear naming, version control support
- **Accessible**: Open-source, CLI-based, standard dependencies  
- **Interoperable**: JSON, Parquet, NPZ, CSV formats
- **Reusable**: Modular design, configurable parameters

