# nmf_plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt

# This constant should ideally be passed from config or main script if it can vary.
# For NMF(random_state=42) without other params, 'frobenius' is the default beta_loss.
DEFAULT_BETA_LOSS_FOR_PLOTTING = 'frobenius'

def plot_k_selection_results(results_df, base_output_dir_for_group, group_prefix):
    """Plots F1, AUPRC, and Reconstruction Error vs. k for a specific group."""
    if results_df.empty:
        print(f"  INFO: No results to plot for {group_prefix}.")
        return

    figures_dir = os.path.join(base_output_dir_for_group, "summary_figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    color_f1 = 'tab:red'
    ax1.set_xlabel('Number of Components (k)', fontsize=12)
    ax1.set_ylabel('Max Mean F1 Score', color=color_f1, fontsize=12)
    ax1.plot(results_df['k'], results_df['max_mean_f1'], color=color_f1, marker='o', linestyle='-', linewidth=2, label='Max Mean F1')
    ax1.tick_params(axis='y', labelcolor=color_f1, labelsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    color_auprc = 'tab:blue'
    ax2.set_ylabel('AUPRC', color=color_auprc, fontsize=12)
    ax2.plot(results_df['k'], results_df['auprc'], color=color_auprc, marker='x', linestyle='--', linewidth=2, label='AUPRC')
    ax2.tick_params(axis='y', labelcolor=color_auprc, labelsize=10)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    color_recon_err = 'tab:green'
    ax3.set_ylabel(f'Reconstruction Error ({DEFAULT_BETA_LOSS_FOR_PLOTTING})', color=color_recon_err, fontsize=12)
    ax3.plot(results_df['k'], results_df['reconstruction_error'], color=color_recon_err, marker='s', linestyle=':', linewidth=2, label=f'Recon. Err. ({DEFAULT_BETA_LOSS_FOR_PLOTTING})')
    ax3.tick_params(axis='y', labelcolor=color_recon_err, labelsize=10)

    lines, labels_leg = ax1.get_legend_handles_labels()
    lines2, labels2_leg = ax2.get_legend_handles_labels()
    lines3, labels3_leg = ax3.get_legend_handles_labels()
    ax1.legend(lines + lines2 + lines3, labels_leg + labels2_leg + labels3_leg, loc='best', fontsize=10)
    
    fig.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout for legend
    min_k_val, max_k_val = results_df["k"].min(), results_df["k"].max()
    plt.title(f'{group_prefix} NMF Eval Metrics vs. k ({min_k_val}-{max_k_val})', fontsize=14, pad=20)
    plt.xticks(results_df['k']) # Show all k values as ticks
    
    plot_path = os.path.join(figures_dir, f"{group_prefix}_k_selection_metrics_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  INFO: Metrics plot for {group_prefix} saved to: {plot_path}")
    plt.close(fig) # Close the figure to free memory

def plot_f1_gradient(results_df, base_output_dir_for_group, group_prefix):
    """Plots the gradient of the Max Mean F1 score for a specific group."""
    if results_df.empty or len(results_df) <= 1 or results_df['max_mean_f1'].notna().sum() <= 1:
        print(f"  INFO: Not enough data points to plot F1 gradient for {group_prefix}.")
        return

    figures_dir = os.path.join(base_output_dir_for_group, "summary_figures")
    os.makedirs(figures_dir, exist_ok=True)
           
    valid_f1_for_grad = results_df.dropna(subset=['max_mean_f1'])
    if len(valid_f1_for_grad) > 1: # Need at least 2 points for gradient
        f1_gradient = np.gradient(valid_f1_for_grad['max_mean_f1'], valid_f1_for_grad['k'])

        fig_grad, ax_grad = plt.subplots(figsize=(12, 7)) # Create a new figure and axes
        ax_grad.plot(valid_f1_for_grad['k'], f1_gradient, marker='o', linestyle='-', color='tab:purple', label='Gradient of Max Mean F1')
        ax_grad.set_xlabel('Number of Components (k)', fontsize=12)
        ax_grad.set_ylabel('Gradient of F1 Score', fontsize=12)
        ax_grad.set_title(f'{group_prefix} Rate of Change of Max Mean F1 Score vs. k', fontsize=14)
        ax_grad.axhline(0, color='grey', linestyle='--', lw=1) # Line at y=0
        ax_grad.grid(True, linestyle=':', alpha=0.6)
        ax_grad.set_xticks(valid_f1_for_grad['k'])
        ax_grad.legend(fontsize=10)
        
        plot_path = os.path.join(figures_dir, f"{group_prefix}_k_selection_f1_gradient_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  INFO: F1 gradient plot for {group_prefix} saved to: {plot_path}")
        plt.close(fig_grad) # Close the figure
    else:
        print(f"  INFO: Not enough valid F1 data points (after NaN drop) to calculate gradient for {group_prefix}.")