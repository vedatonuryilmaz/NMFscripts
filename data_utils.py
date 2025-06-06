import polars as pl
import numpy as np
from scipy.sparse import csr_array, vstack as sp_vstack
import scipy.sparse as sp
from tqdm.auto import tqdm

def get_cancer_type_from_sample_id(sample_id: str) -> str:
    """
    Extracts cancer type from a sample ID with format 'Project-TCGA-...'
    For example, extracts 'ACCx' from 'ACCx-TCGA-OR-A5J2...'.
    """
    if isinstance(sample_id, str) and '-' in sample_id:
        # CORRECTED: Take the first part of the string before the hyphen.
        return sample_id.split('-')[0]
    return "UNKNOWN"

def generate_bool_map(df_lazy_full: pl.LazyFrame, all_tcga_samples: list,
                        n_features_total_for_ranking: int, top_n_features_cutoff: int,
                        bool_map_chunk_size: int) -> sp.csr_matrix:
    """Generates the bool_map (features x samples) from a Polars LazyFrame."""
    print(f"  INFO: Using N_FEATURES_TOTAL_FOR_RANKING = {n_features_total_for_ranking} for ranking.")
    df_ranked_lazy = df_lazy_full.select(
        [(n_features_total_for_ranking - pl.col(sample).rank(method="random", descending=False)).alias(sample)
            for sample in all_tcga_samples]
    )
    print("  INFO: Defined lazy df_ranked expression for bool_map generation.")

    blocks = []
    print(f"  INFO: Generating bool_map with TOP_N_FEATURES_CUTOFF = {top_n_features_cutoff}...")
    for start_idx in tqdm(range(0, n_features_total_for_ranking, bool_map_chunk_size),
                            desc="  Processing feature chunks for bool_map"):
        chunk_data_lazy = df_ranked_lazy.slice(start_idx, bool_map_chunk_size)
        chunk_collected = chunk_data_lazy.collect()
        chunk_bool_numpy = (chunk_collected.to_numpy() < top_n_features_cutoff).astype(np.int8)
        blocks.append(csr_array(chunk_bool_numpy))

    if not blocks:
        raise ValueError("No blocks generated for bool_map.")

    bool_map_sparse_features_x_samples = sp_vstack(blocks, format="csr")
    print(f"  INFO: Generated bool_map_sparse_features_x_samples. Shape: {bool_map_sparse_features_x_samples.shape}, "
            f"Stored elements: {bool_map_sparse_features_x_samples.nnz}")
    return bool_map_sparse_features_x_samples