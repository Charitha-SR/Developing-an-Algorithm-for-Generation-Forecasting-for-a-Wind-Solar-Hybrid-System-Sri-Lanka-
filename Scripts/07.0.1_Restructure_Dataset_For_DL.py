
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import os

# ======================= CONFIG =======================
INPUT_FILE = "data/Sri_Lanka_Wind_Solar_Master(Sample).parquet"
OUTPUT_DIR = "data/dl_features"
HORIZONS = {
    "1h": 1,
    "6h": 6,
    "12h": 12,
    "24h": 24,
    "2d": 48,
    "1week": 168,
    "2week": 336,
    "1month": 720,
}

TARGET_SOURCE_COLS = {
    "Solar": "Solar_Power_Output_KW",
    "Wind": "Wind_Power_Output_KW",
    "Total": "Total_Hybrid_Power_Output_KW",
}

TRAIN_END_YEAR = 2022
# ========================================================


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    day_of_year = df["Datetime"].dt.dayofyear
    days_in_year = np.where(df["Datetime"].dt.is_leap_year, 366, 365)
    df["DayOfYear_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)

    return df


def add_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    for output_name, source_col in TARGET_SOURCE_COLS.items():
        for horizon_name, h in HORIZONS.items():
            df[f"Target_{output_name}_{horizon_name}"] = df[source_col].shift(-h)
    return df


def downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Halves memory usage of numeric columns: float64 -> float32."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def build_features_for_group(district_name: str, solar_size: int, wind_size: int,
                              dataset: ds.Dataset) -> pd.DataFrame:
    table = dataset.to_table(
        filter=(
            (ds.field("District") == district_name)
            & (ds.field("Solar_Panel_Size_KW") == solar_size)
            & (ds.field("Wind_Turbine_Size_KW") == wind_size)
        )
    )
    df = table.to_pandas()

    # CRITICAL: sort by time within this group before any shift-based work
    df = df.sort_values("Datetime").reset_index(drop=True)

    df = add_cyclical_time_features(df)
    df = add_horizon_targets(df)
    df = downcast_floats(df)

    return df


class IncrementalParquetWriter:
    """
    Small helper that opens a ParquetWriter lazily (on the first table it
    receives, so it can infer the schema automatically) and lets you keep
    appending tables to the same file, one group at a time, without ever
    holding the full combined dataset in memory.
    """
    def __init__(self, path: str):
        self.path = path
        self.writer = None
        self.rows_written = 0

    def write(self, df: pd.DataFrame):
        if len(df) == 0:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)
        self.rows_written += len(df)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def main():
    dataset = ds.dataset(INPUT_FILE, format="parquet")

    districts = sorted(
        dataset.to_table(columns=["District"]).column("District").unique().to_pylist()
    )
    solar_sizes = sorted(
        dataset.to_table(columns=["Solar_Panel_Size_KW"]).column("Solar_Panel_Size_KW").unique().to_pylist()
    )
    wind_sizes = sorted(
        dataset.to_table(columns=["Wind_Turbine_Size_KW"]).column("Wind_Turbine_Size_KW").unique().to_pylist()
    )

    total_groups = len(districts) * len(solar_sizes) * len(wind_sizes)
    print(f"Found {len(districts)} districts x {len(solar_sizes)} solar sizes x "
          f"{len(wind_sizes)} wind sizes = {total_groups} groups to process.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train_dataset_dl.parquet")
    test_path = os.path.join(OUTPUT_DIR, "test_dataset_dl.parquet")

    # Remove any previous partial output so we never append onto stale data
    for p in (train_path, test_path):
        if os.path.exists(p):
            os.remove(p)

    train_writer = IncrementalParquetWriter(train_path)
    test_writer = IncrementalParquetWriter(test_path)

    group_num = 0
    for district in districts:
        for solar_size in solar_sizes:
            for wind_size in wind_sizes:
                group_num += 1
                print(f"[{group_num}/{total_groups}] {district} | "
                      f"Solar {solar_size}kW | Wind {wind_size}kW ...", end=" ")

                df = build_features_for_group(district, solar_size, wind_size, dataset)

                if len(df) == 0:
                    print("(no rows - skipped)")
                    continue

                train_mask = df["Year"] <= TRAIN_END_YEAR
                test_mask = df["Year"] > TRAIN_END_YEAR

                train_writer.write(df[train_mask])
                test_writer.write(df[test_mask])

                # df, train_mask, test_mask all go out of scope / get
                # overwritten on the next loop iteration, so Python's
                # garbage collector is free to reclaim this group's
                # memory before we move to the next one.
                print(f"train={train_mask.sum():,} test={test_mask.sum():,}")

    train_writer.close()
    test_writer.close()

    print(f"\nTrain rows written: {train_writer.rows_written:,}  -> {train_path}")
    print(f"Test rows written:  {test_writer.rows_written:,}  -> {test_path}")
    print("\nNote: rows near the end of each group's series will have NaN in the")
    print("longer-horizon Target_* columns (e.g. Target_Total_1month) because")
    print("there's no future data left to shift in. These are handled in the")
    print("training script's data loader, not here.")


if __name__ == "__main__":
    main()