# prepare_data.py
import os
import json
import polars as pl
from scipy.sparse import save_npz
from data_utils import get_cancer_type_from_sample_id, generate_bool_map

def main():
    print("--- Starting Data Preparation ---")
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        print("  INFO: Configuration loaded from config.json")
    except Exception as e:
        print(f"  ERROR: Could not load config.json: {e}")
        return

    TCGA_ZSCORES_PATH = config["TCGA_ZSCORES_PATH"]
    EMB_JSON_PATH = config["EMB_JSON_PATH"]
    PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
    TOP_N_FEATURES_CUTOFF = config["TOP_N_FEATURES_CUTOFF"]
    BOOL_MAP_CHUNK_SIZE = config["BOOL_MAP_CHUNK_SIZE"]
    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
    print(f"  INFO: Preprocessed data will be saved to: {PREPROCESSED_DATA_DIR}")

    print(f"\nStep 1: Loading sample IDs and feature count...")
    try:
        df_schema_scan = pl.scan_parquet(TCGA_ZSCORES_PATH)
        all_tcga_samples = sorted(df_schema_scan.collect_schema().names()[6:])
        n_features_actual = df_schema_scan.select(pl.len()).collect().item()
        print(f"  INFO: Loaded {len(all_tcga_samples)} TCGA sample IDs and found {n_features_actual} total features.")
        with open(os.path.join(PREPROCESSED_DATA_DIR, "all_tcga_samples.json"), 'w') as f: json.dump(all_tcga_samples, f)
        print(f"  SUCCESS: Saved all_tcga_samples.json")
    except Exception as e:
        print(f"  ERROR: Could not load sample info: {e}")
        return

    print(f"\nStep 2: Generating global bool_map...")
    try:
        df_lazy_zscores = pl.scan_parquet(TCGA_ZSCORES_PATH)
        bool_map_sparse = generate_bool_map(
            df_lazy_full=df_lazy_zscores, all_tcga_samples=all_tcga_samples,
            n_features_total_for_ranking=n_features_actual,
            top_n_features_cutoff=TOP_N_FEATURES_CUTOFF,
            bool_map_chunk_size=BOOL_MAP_CHUNK_SIZE
        )
        save_npz(os.path.join(PREPROCESSED_DATA_DIR, "bool_map_overall_sparse_feat_x_sample.npz"), bool_map_sparse)
        print(f"  SUCCESS: Global bool_map saved.")
    except Exception as e:
        print(f"  ERROR: bool_map generation failed: {e}")
        return

    print(f"\nStep 3: Loading and saving embryonic groupings...")
    try:
        with open(EMB_JSON_PATH, 'r') as f:
            embryonic_data = json.load(f)
        with open(os.path.join(PREPROCESSED_DATA_DIR, "emb_groupings.json"), 'w') as f:
            json.dump(embryonic_data, f)
        print(f"  SUCCESS: Loaded and saved embryonic groupings.")
    except Exception as e:
        print(f"  ERROR: Could not load {EMB_JSON_PATH}: {e}")
        return

    print(f"\nStep 4: Creating sample_to_cancer_type map...")
    sample_to_cancer_type_map = {s_id: get_cancer_type_from_sample_id(s_id) for s_id in all_tcga_samples}
    with open(os.path.join(PREPROCESSED_DATA_DIR, "sample_to_cancer_type_map.json"), 'w') as f:
        json.dump(sample_to_cancer_type_map, f)
    print(f"  SUCCESS: Saved sample_to_cancer_type_map.json.")
    print(f"  VERIFICATION: Review the first few mappings:")
    for i, s_id in enumerate(all_tcga_samples[:min(5, len(all_tcga_samples))]):
        print(f"    Sample: {s_id}  -> Mapped Cancer Type: {sample_to_cancer_type_map.get(s_id, 'Not Mapped')}")
        
    print("\n--- Data Preparation Complete ---")

if __name__ == "__main__":
    pl.enable_string_cache() # Corrected Polars API call
    main()