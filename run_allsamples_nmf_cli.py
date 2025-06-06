# run_allsamples_nmf_cli.py
import os
import json
import argparse # For --n-jobs option
import numpy as np
import polars as pl
from scipy.sparse import load_npz
from nmf_workflow import run_nmf_pipeline_for_all_samples

def main_cli_all():
    parser = argparse.ArgumentParser(description="Run NMF K-Optimization for ALL samples.")
    parser.add_argument("--n-jobs", type=int, help="Number of parallel jobs for the k-loop. Overrides config file.")
    args = parser.parse_args()

    print("--- Starting NMF K-Optimization CLI for ALL SAMPLES ---")

    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        print("  INFO: Configuration loaded.")
    except Exception as e:
        print(f"  ERROR: Could not load config.json: {e}")
        return

    PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
    GLOBAL_OUTPUT_DIR = config["GLOBAL_OUTPUT_DIR"]
    K_START, K_END = config["K_RANGE_EMBRYONIC_START"], config["K_RANGE_EMBRYONIC_END"]
    T_START, T_END, T_STEP = config["THRESHOLDS_FOR_EVALUATION_START"], config["THRESHOLDS_FOR_EVALUATION_END"], config["THRESHOLDS_FOR_EVALUATION_STEP"]
    NMF_MAX_ITER_VAL = config.get("NMF_MAX_ITER", 200) # Get from config, default to 200
    
    # Determine n_jobs for parallelism
    n_jobs_to_use = args.n_jobs if args.n_jobs is not None else config.get("N_JOBS_PARALLEL", -1)


    k_range_to_test = range(K_START, K_END + 1)
    thresholds_eval_list = np.round(np.arange(T_START, T_END + T_STEP / 2, T_STEP), 2)

    print(f"\nStep 1: Loading preprocessed data from: {PREPROCESSED_DATA_DIR}...")
    try:
        bool_map_path = os.path.join(PREPROCESSED_DATA_DIR, "bool_map_overall_sparse_feat_x_sample.npz")
        bool_map_overall = load_npz(bool_map_path)
        print(f"  SUCCESS: Loaded global bool_map. Shape: {bool_map_overall.shape}")
    except Exception as e:
        print(f"  ERROR: Required preprocessed file not found. Have you run 'python prepare_data.py' first? Details: {e}")
        return
    
    print("\nStep 2: Calling NMF pipeline function for ALL SAMPLES...")
    run_nmf_pipeline_for_all_samples(
        bool_map_overall=bool_map_overall,
        k_range=k_range_to_test,
        nmf_random_state=config["NMF_RANDOM_STATE"],
        nmf_max_iter_val=NMF_MAX_ITER_VAL, # Pass max_iter
        thresholds=thresholds_eval_list,
        global_output_dir=GLOBAL_OUTPUT_DIR,
        n_jobs=n_jobs_to_use # Use determined n_jobs
    )
    
    print(f"\n--- CLI Script for ALL SAMPLES Finished ---")

if __name__ == "__main__":
    pl.enable_string_cache() # Corrected Polars API call
    main_cli_all()