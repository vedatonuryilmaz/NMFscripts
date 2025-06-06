# run_group_nmf_cli.py
import os
import json
import argparse
import numpy as np
import polars as pl
from scipy.sparse import load_npz
from nmf_workflow import run_nmf_pipeline_for_group

def main_cli():
    parser = argparse.ArgumentParser(description="Run NMF K-Optimization for a specific embryonic group.")
    parser.add_argument("embryonic_group_name", type=str, help="Name of the embryonic group (e.g., 'Ectoderm').")
    parser.add_argument("--n-jobs", type=int, help="Number of parallel jobs for the k-loop. Overrides config file.") # For --n-jobs option
    args = parser.parse_args()
    target_group_name = args.embryonic_group_name

    print(f"--- Starting NMF K-Optimization CLI for: {target_group_name} ---")

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
    NMF_MAX_ITER_VAL = config.get("NMF_MAX_ITER", 200) # Get from config, default to 200 if not found
    
    # Determine n_jobs for parallelism
    n_jobs_to_use = args.n_jobs if args.n_jobs is not None else config.get("N_JOBS_PARALLEL", -1)


    k_range_to_test = range(K_START, K_END + 1)
    thresholds_eval_list = np.round(np.arange(T_START, T_END + T_STEP / 2, T_STEP), 2)

    print(f"\nStep 1: Loading preprocessed data from: {PREPROCESSED_DATA_DIR}...")
    try:
        with open(os.path.join(PREPROCESSED_DATA_DIR, "all_tcga_samples.json"), 'r') as f: all_tcga_samples = json.load(f)
        bool_map_overall = load_npz(os.path.join(PREPROCESSED_DATA_DIR, "bool_map_overall_sparse_feat_x_sample.npz"))
        with open(os.path.join(PREPROCESSED_DATA_DIR, "emb_groupings.json"), 'r') as f: organ_system_groupings = json.load(f)['organ_system_groupings']
        with open(os.path.join(PREPROCESSED_DATA_DIR, "sample_to_cancer_type_map.json"), 'r') as f: sample_to_cancer_type_map = json.load(f)
        print("  SUCCESS: All preprocessed data loaded.")
    except Exception as e:
        print(f"  ERROR: Required preprocessed file not found. Have you run 'python prepare_data.py' first? Details: {e}")
        return

    target_group_info = next((g for g in organ_system_groupings if g['group_name'] == target_group_name), None)
    if not target_group_info:
        valid_groups = [g['group_name'] for g in organ_system_groupings]
        print(f"  ERROR: Embryonic group '{target_group_name}' not found. Available groups are: {valid_groups}")
        return
    
    print(f"\nStep 2: Target group selected: {target_group_name}")
    
    print(f"\nStep 3: Calling NMF pipeline function...")
    run_nmf_pipeline_for_group(
        group_info=target_group_info, all_samples_list=all_tcga_samples,
        sample_to_cancer_type_map=sample_to_cancer_type_map,
        bool_map_overall=bool_map_overall,
        k_range=k_range_to_test, 
        nmf_random_state=config["NMF_RANDOM_STATE"],
        nmf_max_iter_val=NMF_MAX_ITER_VAL, # Pass max_iter
        thresholds=thresholds_eval_list,
        global_output_dir=GLOBAL_OUTPUT_DIR,
        n_jobs=n_jobs_to_use # Use determined n_jobs
    )
    print(f"\n--- CLI Script for {target_group_name} Finished ---")

if __name__ == "__main__":
    pl.enable_string_cache() # Corrected Polars API call
    main_cli()