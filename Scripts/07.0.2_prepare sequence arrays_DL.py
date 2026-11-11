
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import json
import os

# ======================= CONFIG =======================
DATA_DIR = "data/dl_features"
TRAIN_PARQUET = os.path.join(DATA_DIR, "train_dataset_dl.parquet")
TEST_PARQUET = os.path.join(DATA_DIR, "test_dataset_dl.parquet")
OUTPUT_DIR = DATA_DIR   # arrays are saved alongside the parquet files

# How many past hours of context to borrow from train for the start of
# the test period. 168 = 1 week, matching your weekly-lag intuition from
# the earlier ML model. If you later want the GRU to look back further
# than 168 hours, you'd need to re-run this script with a bigger value.
LOOKBACK_HOURS = 168

HORIZONS = {
    "1h": 1, "6h": 6, "12h": 12, "24h": 24,
    "2d": 48, "1week": 168, "2week": 336, "1month": 720,
}
TARGET_SOURCE_COLS = {
    "Solar": "Solar_Power_Output_KW",
    "Wind": "Wind_Power_Output_KW",
    "Total": "Total_Hybrid_Power_Output_KW",
}
# Fixed column order - every array and every downstream script must agree
# on this exact order, so we define it once, here.
FEATURE_COLS = [
    "GHI", "Wind_Speed",
    "Solar_Power_Output_KW", "Wind_Power_Output_KW", "Total_Hybrid_Power_Output_KW",
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos",
]
TARGET_COLS = [
    f"Target_{out_name}_{h_name}"
    for out_name in TARGET_SOURCE_COLS
    for h_name in HORIZONS
]
# ========================================================


def get_category_lists():
    """Reads the full set of District / solar / wind values once, to build
    a fixed, sorted code mapping shared by both train and test."""
    train_ds = ds.dataset(TRAIN_PARQUET, format="parquet")
    districts = sorted(train_ds.to_table(columns=["District"]).column("District").unique().to_pylist())
    solar_sizes = sorted(train_ds.to_table(columns=["Solar_Panel_Size_KW"]).column("Solar_Panel_Size_KW").unique().to_pylist())
    wind_sizes = sorted(train_ds.to_table(columns=["Wind_Turbine_Size_KW"]).column("Wind_Turbine_Size_KW").unique().to_pylist())
    return districts, solar_sizes, wind_sizes


def fetch_group(dataset: ds.Dataset, district, solar_size, wind_size) -> pd.DataFrame:
    table = dataset.to_table(
        filter=(
            (ds.field("District") == district)
            & (ds.field("Solar_Panel_Size_KW") == solar_size)
            & (ds.field("Wind_Turbine_Size_KW") == wind_size)
        )
    )
    df = table.to_pandas()
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


def main():
    print("Reading category lists (District / solar sizes / wind sizes)...")
    districts, solar_sizes, wind_sizes = get_category_lists()
    district_code = {name: i for i, name in enumerate(districts)}
    solar_code = {val: i for i, val in enumerate(solar_sizes)}
    wind_code = {val: i for i, val in enumerate(wind_sizes)}

    with open(os.path.join(OUTPUT_DIR, "category_codes.json"), "w") as f:
        json.dump({
            "district_code": district_code,
            "solar_code": {str(k): v for k, v in solar_code.items()},
            "wind_code": {str(k): v for k, v in wind_code.items()},
        }, f, indent=2)

    n_train_total = pq.ParquetFile(TRAIN_PARQUET).metadata.num_rows
    print(f"Train file has {n_train_total:,} rows total.")

    n_features = len(FEATURE_COLS)
    n_targets = len(TARGET_COLS)

    # ---- Preallocate the on-disk memmap arrays for TRAIN ----
    features_train_path = os.path.join(OUTPUT_DIR, "features_train.dat")
    targets_train_path = os.path.join(OUTPUT_DIR, "targets_train.dat")
    features_train = np.memmap(features_train_path, dtype="float32", mode="w+",
                                shape=(n_train_total, n_features))
    targets_train = np.memmap(targets_train_path, dtype="float32", mode="w+",
                               shape=(n_train_total, n_targets))

    train_ds = ds.dataset(TRAIN_PARQUET, format="parquet")
    test_ds = ds.dataset(TEST_PARQUET, format="parquet")

    # Running accumulators for scaling stats (float64 for precision)
    feat_sum = np.zeros(n_features, dtype=np.float64)
    feat_sumsq = np.zeros(n_features, dtype=np.float64)
    targ_sum = np.zeros(n_targets, dtype=np.float64)
    targ_sumsq = np.zeros(n_targets, dtype=np.float64)
    targ_count = np.zeros(n_targets, dtype=np.float64)

    train_group_meta = []
    test_feature_parts, test_target_parts, test_context_parts = [], [], []
    test_district_parts, test_solar_parts, test_wind_parts = [], [], []

    cursor = 0
    group_num = 0
    total_groups = len(districts) * len(solar_sizes) * len(wind_sizes)

    for district in districts:
        for solar_size in solar_sizes:
            for wind_size in wind_sizes:
                group_num += 1
                print(f"[{group_num}/{total_groups}] {district} | "
                      f"Solar {solar_size}kW | Wind {wind_size}kW ...", end=" ")

                df_train = fetch_group(train_ds, district, solar_size, wind_size)
                if len(df_train) == 0:
                    print("(no train rows - skipped)")
                    continue

                feats = df_train[FEATURE_COLS].to_numpy(dtype=np.float32)
                targs = df_train[TARGET_COLS].to_numpy(dtype=np.float32)
                n = len(df_train)

                features_train[cursor:cursor + n] = feats
                targets_train[cursor:cursor + n] = targs

                # Update scaling accumulators
                feat_sum += feats.sum(axis=0, dtype=np.float64)
                feat_sumsq += (feats.astype(np.float64) ** 2).sum(axis=0)
                valid = ~np.isnan(targs)
                targ_sum += np.nansum(targs, axis=0)
                targ_sumsq += np.nansum(targs.astype(np.float64) ** 2, axis=0)
                targ_count += valid.sum(axis=0)

                d_code, s_code, w_code = district_code[district], solar_code[solar_size], wind_code[wind_size]
                train_group_meta.append({
                    "district_code": d_code, "solar_code": s_code, "wind_code": w_code,
                    "start": cursor, "end": cursor + n,
                })

                # ---- Build this group's TEST array, borrowing context from train ----
                df_test = fetch_group(test_ds, district, solar_size, wind_size)
                if len(df_test) > 0:
                    ctx_len = min(LOOKBACK_HOURS, n)
                    ctx_feats = features_train[cursor + n - ctx_len: cursor + n]
                    ctx_targs = targets_train[cursor + n - ctx_len: cursor + n]

                    test_feats = df_test[FEATURE_COLS].to_numpy(dtype=np.float32)
                    test_targs = df_test[TARGET_COLS].to_numpy(dtype=np.float32)

                    combined_feats = np.concatenate([np.array(ctx_feats), test_feats], axis=0)
                    combined_targs = np.concatenate([np.array(ctx_targs), test_targs], axis=0)
                    is_context = np.concatenate([
                        np.ones(ctx_len, dtype=bool),
                        np.zeros(len(df_test), dtype=bool),
                    ])

                    test_feature_parts.append(combined_feats)
                    test_target_parts.append(combined_targs)
                    test_context_parts.append(is_context)
                    test_district_parts.append(np.full(len(combined_feats), d_code, dtype=np.int32))
                    test_solar_parts.append(np.full(len(combined_feats), s_code, dtype=np.int32))
                    test_wind_parts.append(np.full(len(combined_feats), w_code, dtype=np.int32))

                cursor += n
                print(f"train_rows={n:,} test_rows={len(df_test):,}")

    features_train.flush()
    targets_train.flush()

    print("\nFinalizing scaling statistics (computed from TRAIN data only)...")
    feat_mean = feat_sum / cursor
    feat_var = feat_sumsq / cursor - feat_mean ** 2
    feat_std = np.sqrt(np.clip(feat_var, a_min=1e-8, a_max=None))

    targ_mean = targ_sum / targ_count
    targ_var = targ_sumsq / targ_count - targ_mean ** 2
    targ_std = np.sqrt(np.clip(targ_var, a_min=1e-8, a_max=None))

    with open(os.path.join(OUTPUT_DIR, "scaling_stats.json"), "w") as f:
        json.dump({
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "feature_mean": feat_mean.tolist(),
            "feature_std": feat_std.tolist(),
            "target_mean": targ_mean.tolist(),
            "target_std": targ_std.tolist(),
        }, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "group_metadata_train.json"), "w") as f:
        json.dump({
            "n_rows": cursor,
            "n_features": n_features,
            "n_targets": n_targets,
            "groups": train_group_meta,
        }, f, indent=2)

    print("Concatenating test arrays (small enough to hold directly in RAM)...")
    features_test = np.concatenate(test_feature_parts, axis=0)
    targets_test = np.concatenate(test_target_parts, axis=0)
    test_is_context = np.concatenate(test_context_parts, axis=0)
    test_district_codes = np.concatenate(test_district_parts, axis=0)
    test_solar_codes = np.concatenate(test_solar_parts, axis=0)
    test_wind_codes = np.concatenate(test_wind_parts, axis=0)

    np.save(os.path.join(OUTPUT_DIR, "features_test.npy"), features_test)
    np.save(os.path.join(OUTPUT_DIR, "targets_test.npy"), targets_test)
    np.save(os.path.join(OUTPUT_DIR, "test_is_context.npy"), test_is_context)
    np.save(os.path.join(OUTPUT_DIR, "test_district_codes.npy"), test_district_codes)
    np.save(os.path.join(OUTPUT_DIR, "test_solar_codes.npy"), test_solar_codes)
    np.save(os.path.join(OUTPUT_DIR, "test_wind_codes.npy"), test_wind_codes)

    print(f"\nDone. Train array rows: {cursor:,}")
    print(f"Test array rows (including borrowed context): {len(features_test):,}")
    print(f"  of which real test rows to evaluate: {int((~test_is_context).sum()):,}")
    print(f"All files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()