
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats as scipy_stats
import joblib
import os
import time
from datetime import datetime, timedelta


# ========================== CONFIG ==========================
INPUT_DIR  = "data"
TRAIN_FILE = os.path.join(INPUT_DIR, "train_dataset.parquet")
TEST_FILE  = os.path.join(INPUT_DIR, "test_dataset.parquet")
OUTPUT_DIR = "results/rf_model

FOLDS = [
    {"name": "Fold_1", "train_end": 2013, "test_start": 2014, "test_end": 2016},
    {"name": "Fold_2", "train_end": 2016, "test_start": 2017, "test_end": 2019},
    {"name": "Fold_3", "train_end": 2019, "test_start": 2020, "test_end": 2022},
    {"name": "Fold_4", "train_end": 2022, "test_start": 2023, "test_end": 2025},
]

HORIZONS = {
    "1h": 1, "6h": 6, "12h": 12, "24h": 24,
    "2d": 48, "1week": 168, "2week": 336, "1month": 720,
}

FEATURE_COLS = [
    "GHI", "Wind_Speed", "Hour", "Month", "Day",
    "GHI_lag_1h", "GHI_lag_24h", "GHI_lag_168h",
    "Wind_lag_1h", "Wind_lag_24h", "Wind_lag_168h",
    "Power_lag_1h", "Power_lag_24h", "Power_lag_168h",
]
CATEGORICAL_COL = "District"

# RF hyperparameters (memory-safe)
N_ESTIMATORS     = 100
MAX_DEPTH        = 15
MIN_SAMPLES_LEAF = 5
MAX_SAMPLES      = 0.7
RANDOM_STATE     = 42
N_JOBS           = 4

# Set False to re-run all folds even if already completed
SKIP_COMPLETED   = True
# ============================================================


# ----------------------------------------------------------------
# PROGRESS TRACKER
# ----------------------------------------------------------------
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.done = 0
        self.times = []
        self.start = time.time()

    def record(self, elapsed):
        self.times.append(elapsed)
        self.done += 1

    def skip(self, count=1):
        self.done += count

    def eta(self):
        if not self.times:
            return "calculating after first model..."
        avg = np.mean(self.times)
        left = self.total - self.done
        secs = avg * left
        finish = datetime.now() + timedelta(seconds=secs)
        return (f"{fmt_time(secs)} remaining  |  "
                f"Est. finish: {finish.strftime('%Y-%m-%d %H:%M')}")

    def bar(self, fold_name, horizon):
        pct = (self.done / self.total) * 100
        elapsed = time.time() - self.start
        filled = int(pct / 5)
        bar_str = "█" * filled + "░" * (20 - filled)
        return (f"    [{bar_str}] {self.done}/{self.total} ({pct:.0f}%)  "
                f"Elapsed: {fmt_time(elapsed)}")


def fmt_time(s):
    if s < 60:
        return f"{s:.0f}s"
    elif s < 3600:
        return f"{s / 60:.1f}min"
    else:
        return f"{int(s // 3600)}h {int((s % 3600) // 60)}min"


# ----------------------------------------------------------------
# DATA FUNCTIONS
# ----------------------------------------------------------------
def create_missing_targets(df):
    """Create target columns for 6h, 12h, 24h, 2d if they don't exist."""
    new = {"6h": 6, "12h": 12, "24h": 24, "2d": 48}
    needed = {k: v for k, v in new.items() if f"Target_{k}" not in df.columns}

    if not needed:
        print("    All target columns already exist.\n")
        return df

    print(f"    Creating {len(needed)} new target columns: {list(needed.keys())}")
    for h_name, h_steps in needed.items():
        col = f"Target_{h_name}"
        df[col] = df.groupby("District")["Total_Hybrid_Power_Output_KW"].shift(-h_steps)
        print(f"      + {col}  (shift = -{h_steps})")
    print()
    return df


def prepare_xy(df, target_col, encoder, fit_encoder=False):
    needed = FEATURE_COLS + [CATEGORICAL_COL, target_col]
    d = df[needed].dropna()

    if fit_encoder:
        enc = encoder.fit_transform(d[[CATEGORICAL_COL]])
    else:
        enc = encoder.transform(d[[CATEGORICAL_COL]])

    cols = encoder.get_feature_names_out([CATEGORICAL_COL])
    enc_df = pd.DataFrame(enc, columns=cols, index=d.index)

    X = pd.concat([d[FEATURE_COLS], enc_df], axis=1)
    y = d[target_col]
    return X, y


def compute_metrics(actual, predicted):
    residuals = predicted - actual
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    r2   = r2_score(actual, predicted)

    nonzero = actual != 0
    mape = (np.mean(np.abs((actual[nonzero] - predicted[nonzero])
                           / actual[nonzero])) * 100
            if nonzero.sum() > 0 else np.nan)

    return {
        "RMSE":              round(rmse, 4),
        "MAE":               round(mae, 4),
        "MAPE_%":            round(mape, 2),
        "R2":                round(r2, 6),
        "Residual_Mean":     round(float(np.mean(residuals)), 4),
        "Residual_Std":      round(float(np.std(residuals)), 4),
        "Residual_Skew":     round(float(scipy_stats.skew(residuals)), 4),
        "Residual_Kurtosis": round(float(scipy_stats.kurtosis(residuals)), 4),
    }


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
def main():
    total_start = time.time()
    total_models = len(FOLDS) * len(HORIZONS)
    tracker = ProgressTracker(total_models)

    print("=" * 70)
    print("  RANDOM FOREST - TIME-SERIES CROSS-VALIDATION")
    print(f"  Solar 100kW + Wind 100kW  |  4 folds x 8 horizons = {total_models} models")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- Load data ----
    print(f"\n  [STEP 1] Loading data...")
    print(f"    Train: {TRAIN_FILE}")
    print(f"    Test:  {TEST_FILE}")

    train_df = pd.read_parquet(TRAIN_FILE)
    test_df  = pd.read_parquet(TEST_FILE)
    print(f"    Train: {len(train_df):>10,} rows  (1987-2022)")
    print(f"    Test:  {len(test_df):>10,} rows  (2023-2025)")

    # ---- Combine ----
    print(f"\n  [STEP 2] Combining and preparing...")
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(["District", "Datetime"]).reset_index(drop=True)
    print(f"    Combined: {len(full_df):>10,} rows  (1987-2025)")

    del train_df, test_df

    # ---- Create missing targets ----
    full_df = create_missing_targets(full_df)

    districts = sorted(full_df["District"].unique())
    print(f"    Districts: {len(districts)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_fold_metrics = []

    # ================================================================
    # FOLD LOOP
    # ================================================================
    for fold_idx, fold in enumerate(FOLDS):
        fold_name = fold["name"]
        fold_dir  = os.path.join(OUTPUT_DIR, fold_name)
        model_dir = os.path.join(fold_dir, "models")

        # ---- Auto-resume ----
        eval_path = os.path.join(fold_dir, f"{fold_name}_evaluation.csv")
        if SKIP_COMPLETED and os.path.exists(eval_path):
            print(f"\n  SKIP  {fold_name} - already completed")
            try:
                existing = pd.read_csv(eval_path)
                all_fold_metrics.extend(existing.to_dict("records"))
            except Exception:
                pass
            tracker.skip(count=len(HORIZONS))
            continue

        os.makedirs(model_dir, exist_ok=True)
        fold_start = time.time()

        print(f"\n  {'=' * 66}")
        print(f"  {fold_name}:  Train 1987-{fold['train_end']}  |  "
              f"Test {fold['test_start']}-{fold['test_end']}")
        print(f"  {'=' * 66}")

        fold_train = full_df[full_df["Year"] <= fold["train_end"]]
        fold_test  = full_df[(full_df["Year"] >= fold["test_start"]) &
                             (full_df["Year"] <= fold["test_end"])]

        print(f"  Train: {len(fold_train):>10,}  |  Test: {len(fold_test):>10,}")

        fold_predictions = {}
        fold_metrics = []

        # ============================================================
        # HORIZON LOOP
        # ============================================================
        for h_idx, (h_name, h_steps) in enumerate(HORIZONS.items()):
            target_col = f"Target_{h_name}"
            h_start = time.time()

            # ---- Progress display ----
            print(f"\n{tracker.bar(fold_name, h_name)}")
            print(f"    ETA: {tracker.eta()}")
            print(f"    Now: {fold_name} > {h_name} "
                  f"(horizon {h_idx + 1}/{len(HORIZONS)})")

            # ---- Features ----
            encoder = OneHotEncoder(sparse_output=False,
                                    handle_unknown="ignore")
            X_train, y_train = prepare_xy(fold_train, target_col,
                                          encoder, fit_encoder=True)
            X_test,  y_test  = prepare_xy(fold_test,  target_col,
                                          encoder, fit_encoder=False)
            print(f"    Rows  Train: {len(X_train):>10,}  |  "
                  f"Test: {len(X_test):>10,}")

            # ---- Train ----
            print(f"    Training RF (trees={N_ESTIMATORS}, "
                  f"depth={MAX_DEPTH}, samples={MAX_SAMPLES}) ...",
                  end=" ", flush=True)

            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                max_samples=MAX_SAMPLES,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            )
            model.fit(X_train, y_train)
            train_time = time.time() - h_start
            tracker.record(train_time)
            print(f"done in {fmt_time(train_time)}")

            # ---- Save model ----
            joblib.dump(model, os.path.join(model_dir, f"rf_{h_name}.joblib"))

            # ---- Predict ----
            preds = model.predict(X_test)

            # ---- Collect per district ----
            result = fold_test.loc[X_test.index,
                                   ["District", "Datetime"]].copy()
            result["Horizon"]   = h_name
            result["Actual"]    = y_test.values
            result["Predicted"] = preds
            result["Residual"]  = preds - y_test.values

            for district in result["District"].unique():
                d_res = result[result["District"] == district].copy()
                if district not in fold_predictions:
                    fold_predictions[district] = []
                fold_predictions[district].append(d_res)

            # ---- Metrics ----
            metrics = compute_metrics(y_test.values, preds)
            metrics["Fold"]       = fold_name
            metrics["Horizon"]    = h_name
            metrics["Train_Rows"] = len(X_train)
            metrics["Test_Rows"]  = len(X_test)
            fold_metrics.append(metrics)

            print(f"    >>> RMSE: {metrics['RMSE']:.4f}  |  "
                  f"MAE: {metrics['MAE']:.4f}  |  "
                  f"MAPE: {metrics['MAPE_%']:.2f}%  |  "
                  f"R2: {metrics['R2']:.6f}")

            # ---- Per-district metrics ----
            d_rows = []
            for district in sorted(result["District"].unique()):
                d_data = result[result["District"] == district]
                d_m = compute_metrics(d_data["Actual"].values,
                                      d_data["Predicted"].values)
                d_m["District"] = district
                d_m["Horizon"]  = h_name
                d_rows.append(d_m)
            pd.DataFrame(d_rows).to_csv(
                os.path.join(fold_dir, f"district_metrics_{h_name}.csv"),
                index=False)

            # ---- Feature importances ----
            imp = (pd.Series(model.feature_importances_,
                             index=X_train.columns)
                   .sort_values(ascending=False))
            imp.to_csv(
                os.path.join(fold_dir, f"feature_importances_{h_name}.csv"),
                header=["Importance"])
            print(f"    Top 3: {imp.index[0]} ({imp.iloc[0]:.3f}), "
                  f"{imp.index[1]} ({imp.iloc[1]:.3f}), "
                  f"{imp.index[2]} ({imp.iloc[2]:.3f})")

        # ---- Save Excel ----
        print(f"\n    Saving Excel (25 district sheets) ...",
              end=" ", flush=True)
        excel_path = os.path.join(fold_dir, f"{fold_name}_predictions.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for district in sorted(fold_predictions.keys()):
                d_all = pd.concat(fold_predictions[district],
                                  ignore_index=True)
                d_all = d_all.sort_values(["Datetime", "Horizon"])
                d_all.to_excel(writer, sheet_name=district[:31],
                               index=False)
        print("done")
        print(f"    -> {excel_path}")

        # ---- Save evaluation ----
        fold_df = pd.DataFrame(fold_metrics)
        fold_df.to_csv(eval_path, index=False)
        print(f"    -> {eval_path}")

        # ---- Save residuals ----
        res_parts = []
        for d_list in fold_predictions.values():
            for d_df in d_list:
                res_parts.append(d_df[["District", "Horizon", "Residual"]])
        pd.concat(res_parts, ignore_index=True).to_csv(
            os.path.join(fold_dir, f"{fold_name}_residuals.csv"),
            index=False)
        print(f"    -> {fold_name}_residuals.csv")

        all_fold_metrics.extend(fold_metrics)

        fold_time = time.time() - fold_start
        print(f"\n    >>> {fold_name} completed in {fmt_time(fold_time)}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  CROSS-VALIDATION SUMMARY - ALL 4 FOLDS")
    print(f"{'=' * 70}")

    summary_df = pd.DataFrame(all_fold_metrics)
    summary_path = os.path.join(OUTPUT_DIR, "CV_summary_all_folds.csv")
    summary_df.to_csv(summary_path, index=False)

    # ---- Per-horizon table ----
    hdr = f"  {'Horizon':<10} {'Fold':<10} {'RMSE':>10} {'MAE':>10} {'MAPE%':>10} {'R2':>12}"
    for h_name in HORIZONS.keys():
        h_data = summary_df[summary_df["Horizon"] == h_name]
        if len(h_data) == 0:
            continue
        print(f"\n  --- {h_name} ---")
        print(hdr)
        print(f"  {'_' * 64}")
        for _, row in h_data.iterrows():
            print(f"  {h_name:<10} {row['Fold']:<10} "
                  f"{row['RMSE']:>10.4f} {row['MAE']:>10.4f} "
                  f"{row['MAPE_%']:>10.2f} {row['R2']:>12.6f}")
        print(f"  {'MEAN':<10} {'':>10} "
              f"{h_data['RMSE'].mean():>10.4f} "
              f"{h_data['MAE'].mean():>10.4f} "
              f"{h_data['MAPE_%'].mean():>10.2f} "
              f"{h_data['R2'].mean():>12.6f}")

    # ---- Overall fold comparison ----
    print(f"\n  --- Average across all horizons ---")
    print(f"  {'Fold':<10} {'Avg RMSE':>10} {'Avg MAE':>10} "
          f"{'Avg MAPE%':>10} {'Avg R2':>12}")
    print(f"  {'_' * 56}")
    for fold in FOLDS:
        f_data = summary_df[summary_df["Fold"] == fold["name"]]
        if len(f_data) == 0:
            continue
        print(f"  {fold['name']:<10} "
              f"{f_data['RMSE'].mean():>10.4f} "
              f"{f_data['MAE'].mean():>10.4f} "
              f"{f_data['MAPE_%'].mean():>10.2f} "
              f"{f_data['R2'].mean():>12.6f}")

    print(f"\n  Summary saved -> {summary_path}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  TOTAL TIME: {fmt_time(total_time)}")
    print(f"  MODELS COMPLETED: {tracker.done}/{tracker.total}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()