#!/bin/bash
# A script to orchestrate NMF runs for all embryonic groups and the full dataset.

# --- OPTION 1: Sequential Execution (Safest & Recommended) ---
# This runs one group after another, ensuring no resource contention.
# Each individual run can use all available cores (n_jobs=-1 from config).
# This is the most reliable way to run the full analysis.
echo "INFO: Starting sequential execution of all NMF analyses..."

echo "--- Running Ectoderm ---"
python run_group_nmf_cli.py Ectoderm && \
echo "--- Running Mesoderm ---" && \
python run_group_nmf_cli.py Mesoderm && \
echo "--- Running Endoderm ---" && \
python run_group_nmf_cli.py Endoderm && \
echo "--- Running All Samples ---" && \
python run_allsamples_nmf_cli.py && \

echo "SUCCESS: All sequential analyses finished."


# --- OPTION 2: Concurrent Execution (Advanced - Use with Caution) ---
# This runs all groups at the same time in the background.
# You MUST limit the cores each process uses to avoid system overload.
#
# To use this option, comment out the "Sequential Execution" block above and
# uncomment the block below.
#
# echo "INFO: Starting concurrent execution of NMF analyses..."
# # Example: If you have 16 cores, you could give 5 to each of the 3 groups.
# CORES_PER_PROCESS=5
#
# echo "Launching Ectoderm with $CORES_PER_PROCESS cores..."
# python run_group_nmf_cli.py Ectoderm --n-jobs $CORES_PER_PROCESS &
#
# echo "Launching Mesoderm with $CORES_PER_PROCESS cores..."
# python run_group_nmf_cli.py Mesoderm --n-jobs $CORES_PER_PROCESS &
#
# echo "Launching Endoderm with $CORES_PER_PROCESS cores..."
# python run_group_nmf_cli.py Endoderm --n-jobs $CORES_PER_PROCESS &
#
# echo "All concurrent jobs launched. Monitor with 'top' or 'htop'."
# echo "Waiting for all background jobs to complete before continuing..."
# wait
# echo "Concurrent embryonic group analyses finished."
#
# echo "--- Running All Samples (after concurrent jobs finish) ---"
# python run_allsamples_nmf_cli.py
#
# echo "SUCCESS: All analyses finished."