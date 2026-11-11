

import pandas as pd
import pyarrow.dataset as ds
import os

# ======================= CONFIG =======================
INPUT_FILE = "data/Sri_Lanka_Wind_Solar_Master(Sample).parquet"
OUTPUT_DIR = "data"

SOLAR_SIZE = 100   # kW 
WIND_SIZE = 100    # kW

LAG_HOURS = [1, 24, 168]                       # past 1h, 1 day, 1 week
HORIZONS = {"1h": 1, "1week": 168, "2week": 336, "1month": 720}  # future targets to create

TRAIN_END_YEAR = 2022   # train on data through this year
# test = TRAIN_END_YEAR+1 .. end of dataset
# ========================================================


def build_features_for_district(district_name: str, dataset: ds.Dataset) -> pd.DataFrame:
    table = dataset.to_table(
        filter=(
            (ds.field("District") == district_name)
            & (ds.field("Solar_Panel_Size_KW") == SOLAR_SIZE)
            & (ds.field("Wind_Turbine_Size_KW") == WIND_SIZE)
        )
    )
    df = table.to_pandas()
    df = df.sort_values("Datetime").reset_index(drop=True)

    # ---- Lag features: what the model is ALLOWED to know (past values) ----
    for lag in LAG_HOURS:
        df[f"GHI_lag_{lag}h"] = df["GHI"].shift(lag)
        df[f"Wind_lag_{lag}h"] = df["Wind_Speed"].shift(lag)
        df[f"Power_lag_{lag}h"] = df["Total_Hybrid_Power_Output_KW"].shift(lag)

    # ---- Horizon targets: what the model must PREDICT (future values) ----
    for name, h in HORIZONS.items():
        df[f"Target_{name}"] = df["Total_Hybrid_Power_Output_KW"].shift(-h)

    return df


def main():
    dataset = ds.dataset(INPUT_FILE, format="parquet")
    districts = sorted(
        dataset.to_table(columns=["District"]).column("District").unique().to_pylist()
    )
    print(f"Building features for {len(districts)} districts "
          f"(combination: {SOLAR_SIZE}kW solar + {WIND_SIZE}kW wind)...\n")

    train_parts, test_parts = [], []

    for i, district in enumerate(districts, start=1):
        print(f"[{i}/{len(districts)}] {district} ...")
        df = build_features_for_district(district, dataset)

        train_mask = df["Year"] <= TRAIN_END_YEAR
        test_mask = df["Year"] > TRAIN_END_YEAR

        train_parts.append(df[train_mask])
        test_parts.append(df[test_mask])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train_dataset.parquet")
    test_path = os.path.join(OUTPUT_DIR, "test_dataset.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nTrain rows: {len(train_df):,}  -> saved to {train_path}")
    print(f"Test rows:  {len(test_df):,}  -> saved to {test_path}")
    print("\nNote: rows near the start of each district's series (missing lag_168h)")
    print("and near the end (missing Target_1month) will contain NaNs - these are")
    print("dropped automatically in the training scripts, not here.")


if __name__ == "__main__":
    main()