# nmf_workflow.py
import os
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import issparse
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from nmf_evaluation import calculate_nmf_evaluation_metrics
from nmf_plotting import plot_k_selection_results, plot_f1_gradient

def _run_nmf_for_single_k(k_val, X_input_subset, nmf_random_state, nmf_max_iter,
                            output_weights_dir_for_k, V_original_dense_subset,
                            thresholds_eval, tmp_results_dir):
    """Helper function to run NMF for a single k and save its result immediately."""
    try:
        # Using specified max_iter. Other NMF params use scikit-learn defaults
        # for NMF(random_state=X) which are:
        # init=None (often 'nndsvda' for non-negative X, or 'nndsvd' if H and W are provided for init)
        # solver='cd' (for frobenius loss)
        # beta_loss='frobenius'
        # tol=1e-4
        # l1_ratio=0
        # alpha_W=0.0 (formerly alpha)
        # alpha_H='same'
        model = NMF(n_components=k_val,
                    random_state=nmf_random_state,
                    max_iter=nmf_max_iter)
        
        H_matrix = model.fit_transform(X_input_subset)
        W_matrix = model.components_

        os.makedirs(output_weights_dir_for_k, exist_ok=True)
        np.save(os.path.join(output_weights_dir_for_k, f"W_k{k_val}.npy"), W_matrix.T)
        np.save(os.path.join(output_weights_dir_for_k, f"H_k{k_val}.npy"), H_matrix)
        
        R_reconstructed = H_matrix @ W_matrix
        max_f1, auprc = calculate_nmf_evaluation_metrics(V_original_dense_subset, R_reconstructed, thresholds_eval)
        
        result_dict = {
            'k': k_val,
            'max_mean_f1': max_f1,
            'auprc': auprc,
            'reconstruction_error': model.reconstruction_err_,
            'nmf_params': { # Logging NMF parameters used for this k
                'n_components': k_val,
                'random_state': nmf_random_state,
                'max_iter': nmf_max_iter,
                'solver': model.solver, # Actual solver used by sklearn
                'beta_loss': model.beta_loss, # Actual beta_loss used
                'init': model.init if model.init is not None else 'auto (sklearn default)', # Actual init used
                'tol': model.tol # Actual tolerance
            }
        }
        
        with open(os.path.join(tmp_results_dir, f"result_k_{k_val}.json"), 'w') as f:
            json.dump(result_dict, f, indent=2) # indent for readability
            
        return result_dict
    except Exception as e:
        print(f"  ERROR in NMF for k={k_val}: {e}")
        return None

def _execute_nmf_loop(X_nmf_input, k_range, nmf_random_state, nmf_max_iter_val, thresholds,
                      base_output_dir, group_prefix, n_jobs):
    """Shared logic for running the NMF loop, plotting, and returning results."""
    tmp_results_dir = os.path.join(base_output_dir, "tmp_results")
    os.makedirs(tmp_results_dir, exist_ok=True)
    
    V_original_dense = X_nmf_input.toarray() if issparse(X_nmf_input) else np.array(X_nmf_input)
    
    tasks = []
    for k_val_iter in k_range:
        k_weights_dir = os.path.join(base_output_dir, f"{k_val_iter}NMF", "weights")
        tasks.append(delayed(_run_nmf_for_single_k)(
            k_val_iter, X_nmf_input, nmf_random_state, nmf_max_iter_val, k_weights_dir,
            V_original_dense, thresholds, tmp_results_dir
        ))

    effective_n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
    print(f"  INFO: Running NMF for k in {list(k_range)} using up to {effective_n_jobs} parallel jobs.")
    print(f"  INFO: NMF Parameters: random_state={nmf_random_state}, max_iter={nmf_max_iter_val}, other params: scikit-learn defaults.")
    
    with tqdm(total=len(tasks), desc=f"  NMF k-loop for {group_prefix}") as pbar:
        results_from_parallel_raw = Parallel(n_jobs=n_jobs)(
            (pbar.update(1) or task) for task in tasks
        )
    
    all_results = [res for res in results_from_parallel_raw if res is not None]

    if not all_results:
        print(f"  WARNING: No NMF results were successfully obtained for {group_prefix}. Check for errors.")
        if os.path.exists(tmp_results_dir): shutil.rmtree(tmp_results_dir)
        return None

    results_df = pd.DataFrame(all_results).sort_values(by='k').reset_index(drop=True)
    csv_path = os.path.join(base_output_dir, f"{group_prefix}_nmf_evaluation_summary.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  INFO: Evaluation summary for {group_prefix} saved to: {csv_path}")
    
    # Save a separate file with just the NMF parameters used for this entire run for clarity
    run_params_log = {
        'group': group_prefix,
        'k_range': list(k_range),
        'nmf_random_state': nmf_random_state,
        'nmf_max_iter': nmf_max_iter_val,
        'thresholds_count': len(thresholds),
        'input_matrix_shape': X_nmf_input.shape,
        'sklearn_nmf_defaults_note': "NMF uses scikit-learn defaults for init, solver, beta_loss, tol unless overridden."
    }
    with open(os.path.join(base_output_dir, f"{group_prefix}_nmf_run_parameters.json"), 'w') as f:
        json.dump(run_params_log, f, indent=2)
    print(f"  INFO: Overall NMF run parameters logged for {group_prefix}.")


    print(f"  INFO: Generating plots for {group_prefix}...")
    plot_k_selection_results(results_df, base_output_dir, group_prefix)
    plot_f1_gradient(results_df, base_output_dir, group_prefix)
    
    if os.path.exists(tmp_results_dir): shutil.rmtree(tmp_results_dir)
    print(f"=== Finished NMF Pipeline for: {group_prefix} ===")
    return results_df

def run_nmf_pipeline_for_group(group_info, all_samples_list, sample_to_cancer_type_map,
                               bool_map_overall, k_range, nmf_random_state, nmf_max_iter_val, thresholds,
                               global_output_dir, n_jobs):
    """Subsets data for a group and runs the NMF pipeline."""
    group_name = group_info['group_name']
    group_prefix = group_name[:3].capitalize()
    min_k, max_k = min(k_range), max(k_range)
    print(f"\n=== Preparing NMF Pipeline for Embryonic Group: {group_name} ({group_prefix}) ===")

    group_base_output_dir = os.path.join(global_output_dir, f"{group_prefix}_NMF_K_opt_{min_k}_{max_k}")
    
    group_cancer_codes = set(group_info['cancer_codes'])
    group_sample_indices = [
        idx for idx, sample_id in enumerate(all_samples_list)
        if sample_to_cancer_type_map.get(sample_id) in group_cancer_codes
    ]

    if not group_sample_indices:
        print(f"  WARNING: No samples found for {group_name}. Skipping.")
        return None
    print(f"  INFO: Found {len(group_sample_indices)} samples for group {group_name}.")

    X_nmf_input_group = bool_map_overall[:, group_sample_indices].T.astype(np.float64)
    
    if X_nmf_input_group.shape[0] < min(k_range):
        print(f"  WARNING: Not enough samples ({X_nmf_input_group.shape[0]}) for min_k={min(k_range)}. Skipping.")
        return None
    print(f"  INFO: NMF input matrix for {group_name} (samples x features): {X_nmf_input_group.shape}")

    return _execute_nmf_loop(X_nmf_input_group, k_range, nmf_random_state, nmf_max_iter_val, thresholds,
                               group_base_output_dir, group_prefix, n_jobs)

def run_nmf_pipeline_for_all_samples(bool_map_overall, k_range, nmf_random_state, nmf_max_iter_val, thresholds,
                                     global_output_dir, n_jobs):
    """Runs the NMF pipeline on the full dataset without subsetting."""
    group_prefix = "AllSamples"
    min_k, max_k = min(k_range), max(k_range)
    print(f"\n=== Preparing NMF Pipeline for ALL SAMPLES ({group_prefix}) ===")
    
    base_output_dir = os.path.join(global_output_dir, f"{group_prefix}_NMF_K_opt_{min_k}_{max_k}")
    X_nmf_input = bool_map_overall.T.astype(np.float64)
    print(f"  INFO: NMF input matrix for All Samples (samples x features): {X_nmf_input.shape}")

    return _execute_nmf_loop(X_nmf_input, k_range, nmf_random_state, nmf_max_iter_val, thresholds,
                               base_output_dir, group_prefix, n_jobs)