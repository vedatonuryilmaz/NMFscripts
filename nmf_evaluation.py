# nmf_evaluation.py
import numpy as np
from sklearn.metrics import auc

def calculate_nmf_evaluation_metrics(
    original_binary_matrix_transposed, # samples x features
    reconstructed_continuous_matrix_transposed, # samples x features
    thresholds_list
):
    """Calculates Max Mean F1 and AUPRC."""
    num_samples_eval_func = original_binary_matrix_transposed.shape[0]
    mean_f1s_at_thresholds_func = []
    overall_precisions_at_thresholds_func = []
    overall_recalls_at_thresholds_func = []

    original_binary_matrix_bool_func = original_binary_matrix_transposed.astype(bool)

    for thresh_eval_func in thresholds_list:
        f1_scores_for_this_thresh_samples_func = []
        tp_total_for_thresh_func, fp_total_for_thresh_func, fn_total_for_thresh_func = 0, 0, 0

        for i_func in range(num_samples_eval_func):
            original_bin_sample_func = original_binary_matrix_bool_func[i_func, :]
            reconstructed_bin_sample_func = reconstructed_continuous_matrix_transposed[i_func, :] > thresh_eval_func
            
            tp_sample_func = np.sum(original_bin_sample_func & reconstructed_bin_sample_func)
            fp_sample_func = np.sum(~original_bin_sample_func & reconstructed_bin_sample_func)
            fn_sample_func = np.sum(original_bin_sample_func & ~reconstructed_bin_sample_func)
            
            tp_total_for_thresh_func += tp_sample_func
            fp_total_for_thresh_func += fp_sample_func
            fn_total_for_thresh_func += fn_sample_func
            
            precision_s_func = tp_sample_func / (tp_sample_func + fp_sample_func) if (tp_sample_func + fp_sample_func) > 0 else 0
            recall_s_func = tp_sample_func / (tp_sample_func + fn_sample_func) if (tp_sample_func + fn_sample_func) > 0 else 0
            f1_s_func = 2 * (precision_s_func * recall_s_func) / (precision_s_func + recall_s_func) if (precision_s_func + recall_s_func) > 0 else 0
            f1_scores_for_this_thresh_samples_func.append(f1_s_func)

        mean_f1s_at_thresholds_func.append(np.mean(f1_scores_for_this_thresh_samples_func) if f1_scores_for_this_thresh_samples_func else 0.0)
        precision_overall_thresh_func = tp_total_for_thresh_func / (tp_total_for_thresh_func + fp_total_for_thresh_func) if (tp_total_for_thresh_func + fp_total_for_thresh_func) > 0 else 0
        recall_overall_thresh_func = tp_total_for_thresh_func / (tp_total_for_thresh_func + fn_total_for_thresh_func) if (tp_total_for_thresh_func + fn_total_for_thresh_func) > 0 else 0
        overall_precisions_at_thresholds_func.append(precision_overall_thresh_func)
        overall_recalls_at_thresholds_func.append(recall_overall_thresh_func)

    max_mean_f1_for_k_func = np.max(mean_f1s_at_thresholds_func) if mean_f1s_at_thresholds_func else 0.0

    recall_points_auprc_func = np.concatenate(([0.0], overall_recalls_at_thresholds_func, [1.0]))
    first_precision_point_func = overall_precisions_at_thresholds_func[0] if overall_precisions_at_thresholds_func else 0.5 
    precision_points_auprc_func = np.concatenate(([first_precision_point_func], overall_precisions_at_thresholds_func, [0.0]))
    
    sorted_indices_auprc_func = np.argsort(recall_points_auprc_func)
    recall_points_sorted_func = recall_points_auprc_func[sorted_indices_auprc_func]
    precision_points_sorted_func = precision_points_auprc_func[sorted_indices_auprc_func]
    
    auprc_for_k_func = 0.0
    if len(recall_points_sorted_func) > 1 and len(np.unique(recall_points_sorted_func)) > 1:
        try:
            auprc_for_k_func = auc(recall_points_sorted_func, precision_points_sorted_func)
        except ValueError: # Handle cases like all precisions being zero
            auprc_for_k_func = 0.0
            
    return max_mean_f1_for_k_func, auprc_for_k_func