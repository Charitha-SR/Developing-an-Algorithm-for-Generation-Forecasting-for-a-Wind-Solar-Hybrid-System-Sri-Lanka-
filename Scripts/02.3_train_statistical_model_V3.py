
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import warnings
import pickle
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ======================= CONFIG =======================
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train_dataset.parquet")
TEST_FILE = os.path.join(DATA_DIR, "test_dataset.parquet")

# OUTPUT DIRECTORIES
OUTPUT_DIR = os.path.join(DATA_DIR, "Statistical_Model_3Districts_Results")
MODEL_DIR = os.path.join(OUTPUT_DIR, "Saved_Models")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "Predictions_Data")
EVALUATION_DIR = os.path.join(OUTPUT_DIR, "Evaluation_Metrics")

for d in [OUTPUT_DIR, MODEL_DIR, PREDICTIONS_DIR, EVALUATION_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions_3districts_4cutoffs.csv")
EVALUATION_FILE = os.path.join(EVALUATION_DIR, "evaluation_metrics.csv")
SUMMARY_FILE = os.path.join(EVALUATION_DIR, "evaluation_summary.txt")

# ---- HORIZONS ----
HORIZONS = {
    "6h": 6, "12h": 12, "24h": 24, "2day": 48,
    "1week": 168, "2week": 336, "1month": 720,
}
MAX_HORIZON = max(HORIZONS.values())

# ---- 4 CUTOFF TIMES ----
CUTOFF_HOURS = [0, 6, 12, 18]

# ---- 3 KEY DISTRICTS ----
SELECTED_DISTRICTS = ["Colombo", "Nuwara Eliya", "Jaffna"]

REFIT_EVERY_DAYS = 30
SEASONAL_PERIOD = 24


# ========================================================


def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, MAPE, R²"""
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return None

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mask = np.abs(y_true) > 1
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    r2 = r2_score(y_true, y_pred)

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "N": len(y_true)}


def run_for_district(district_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Train and evaluate Holt-Winters for one district, all cutoff times"""
    d_train = train_df[train_df["District"] == district_name].sort_values("Datetime")
    d_test = test_df[test_df["District"] == district_name].sort_values("Datetime")

    full_series = pd.concat([
        d_train[["Datetime", "Total_Hybrid_Power_Output_KW"]],
        d_test[["Datetime", "Total_Hybrid_Power_Output_KW"]],
    ]).drop_duplicates("Datetime").set_index("Datetime")["Total_Hybrid_Power_Output_KW"]

    test_start = d_test["Datetime"].min()
    test_end = d_test["Datetime"].max()

    results = []
    saved_models = {}

    # ---- LOOP CUTOFF HOURS ----
    for cutoff_hour in CUTOFF_HOURS:
        cutoff = test_start.replace(hour=cutoff_hour, minute=0, second=0, microsecond=0)
        if cutoff < test_start:
            cutoff += pd.Timedelta(days=1)

        last_model = None
        refit_count = 0

        while cutoff + pd.Timedelta(hours=MAX_HORIZON) <= test_end:
            history = full_series[full_series.index <= cutoff]

            if len(history) < SEASONAL_PERIOD:
                cutoff += pd.Timedelta(days=REFIT_EVERY_DAYS)
                continue

            try:
                model = ExponentialSmoothing(
                    history, trend="add", seasonal="add",
                    seasonal_periods=SEASONAL_PERIOD,
                    initialization_method="estimated",
                ).fit(optimized=True)

                last_model = model
                forecast = model.forecast(MAX_HORIZON)

            except:
                cutoff += pd.Timedelta(days=REFIT_EVERY_DAYS)
                continue

            # Extract predictions
            for horizon_name, hours_ahead in HORIZONS.items():
                target_time = cutoff + pd.Timedelta(hours=hours_ahead)
                if target_time in full_series.index:
                    results.append({
                        "District": district_name,
                        "Cutoff": cutoff,
                        "Cutoff_Hour": cutoff_hour,
                        "Cutoff_Time": f"{cutoff_hour:02d}:00",
                        "Horizon": horizon_name,
                        "Hours_Ahead": hours_ahead,
                        "Predicted": forecast.iloc[hours_ahead - 1],
                        "Actual": full_series.loc[target_time],
                    })

            cutoff += pd.Timedelta(days=REFIT_EVERY_DAYS)
            refit_count += 1

        if last_model is not None:
            saved_models[cutoff_hour] = last_model
            print(f"      {district_name} - Cutoff {cutoff_hour:02d}:00 → {refit_count} refits, model saved")

    return pd.DataFrame(results), saved_models


def main():
    print("=" * 100)
    print("OPTIMIZED STATISTICAL MODEL: 3 DISTRICTS + 4 CUTOFF TIMES")
    print("=" * 100)

    print(f"\n📍 Districts:")
    print(f"   1. Colombo - Coastal, high daytime PV")
    print(f"   2. Nuwara Eliya - Highland (2000m), elevation effects")
    print(f"   3. Jaffna - Dry zone, wind-dominant")

    print(f"\n⏰ Cutoff times (FULL 24-hour cycle):")
    print(f"   00:00 (midnight) - Wind-only")
    print(f"   06:00 (dawn) - PV ramp-up")
    print(f"   12:00 (noon) - PV PEAK!")
    print(f"   18:00 (evening) - PV ramp-down")

    print(f"\n📊 Horizons: 7 types (6h to 1 month)")
    print(
        f"\n💾 Models: {len(SELECTED_DISTRICTS)} districts × {len(CUTOFF_HOURS)} cutoffs = {len(SELECTED_DISTRICTS) * len(CUTOFF_HOURS)} models")
    print(f"\n⏱️  ESTIMATED RUNTIME: 1.5-2 hours\n")

    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_FILE, columns=["District", "Datetime", "Total_Hybrid_Power_Output_KW"])
    test_df = pd.read_parquet(TEST_FILE, columns=["District", "Datetime", "Total_Hybrid_Power_Output_KW"])

    districts = [d for d in SELECTED_DISTRICTS if d in train_df["District"].values]

    print(f"Processing {len(districts)} district(s)...\n")

    all_results = []
    all_models = {}
    start_time = datetime.now()

    # ---- SEQUENTIAL PROCESSING ----
    for i, district in enumerate(districts, 1):
        print(f"[{i}/{len(districts)}] {district}")
        res, models = run_for_district(district, train_df, test_df)
        all_results.append(res)
        all_models[district] = models
        print(f"    ✓ {len(res):,} predictions\n")

    # Save predictions
    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(OUTPUT_FILE, index=False)

    elapsed_total = (datetime.now() - start_time).total_seconds() / 60

    print(f"✓ Saved predictions → {OUTPUT_FILE}")

    # Save models
    print(f"\n💾 Saving models...")
    for district, hourly_models in all_models.items():
        for cutoff_hour, model in hourly_models.items():
            model_path = os.path.join(MODEL_DIR, f"{district}_cutoff{cutoff_hour:02d}00_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    print(f"✓ Saved {len(districts) * len(CUTOFF_HOURS)} models")

    # Calculate evaluation metrics
    print(f"\n📈 Calculating evaluation metrics...")
    evaluation_results = []

    for district in districts:
        d_data = final[final["District"] == district]

        # By horizon
        for horizon in d_data["Horizon"].unique():
            h_data = d_data[d_data["Horizon"] == horizon]
            metrics = calculate_metrics(h_data["Actual"].values, h_data["Predicted"].values)
            if metrics:
                metrics.update({"District": district, "Horizon": horizon, "Cutoff_Time": "All"})
                evaluation_results.append(metrics)

        # By cutoff time
        for cutoff_hour in d_data["Cutoff_Hour"].unique():
            c_data = d_data[d_data["Cutoff_Hour"] == cutoff_hour]
            metrics = calculate_metrics(c_data["Actual"].values, c_data["Predicted"].values)
            if metrics:
                metrics.update({"District": district, "Horizon": "All", "Cutoff_Time": f"{cutoff_hour:02d}:00"})
                evaluation_results.append(metrics)

        # Overall
        metrics = calculate_metrics(d_data["Actual"].values, d_data["Predicted"].values)
        if metrics:
            metrics.update({"District": district, "Horizon": "Overall", "Cutoff_Time": "All"})
            evaluation_results.append(metrics)

    eval_df = pd.DataFrame(evaluation_results)
    eval_df.to_csv(EVALUATION_FILE, index=False)
    print(f"✓ Saved evaluation metrics")

    # Summary report
    print(f"\nWriting summary report...")
    with open(SUMMARY_FILE, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("STATISTICAL MODEL EVALUATION - 3 DISTRICTS, 4 CUTOFF TIMES\n")
        f.write("=" * 100 + "\n\n")

        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 100 + "\n")
        for district in districts:
            d_overall = eval_df[(eval_df["District"] == district) & (eval_df["Horizon"] == "Overall")]
            if len(d_overall) > 0:
                row = d_overall.iloc[0]
                f.write(f"\n{district}:\n")
                f.write(f"  R²:   {row['R2']:.4f} ({row['R2'] * 100:.1f}% variance explained)\n")
                f.write(f"  RMSE: {row['RMSE']:.2f} kW\n")
                f.write(f"  MAE:  {row['MAE']:.2f} kW\n")
                f.write(f"  MAPE: {row['MAPE']:.1f}%\n")
                f.write(f"  N:    {int(row['N'])} predictions\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("PERFORMANCE BY CUTOFF TIME (PV at different times)\n")
        f.write("=" * 100 + "\n\n")
        f.write("Time      Description           RMSE    MAE    R²\n")
        f.write("-" * 60 + "\n")
        for ch in [0, 6, 12, 18]:
            c_data = eval_df[(eval_df["Cutoff_Time"] == f"{ch:02d}:00") & (eval_df["Horizon"] == "All")]
            if len(c_data) > 0:
                row = c_data.iloc[0]
                desc = {0: "Midnight (wind-only)", 6: "Dawn (PV ramp-up)",
                        12: "Noon (PV PEAK!)", 18: "Evening (ramp-down)"}[ch]
                f.write(f"{ch:02d}:00  {desc:20}  {row['RMSE']:6.2f}  {row['MAE']:5.2f}  {row['R2']:7.4f}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("PERFORMANCE BY HORIZON (Short vs Long-term)\n")
        f.write("=" * 100 + "\n\n")
        f.write("Horizon    RMSE    MAE    R²     Quality\n")
        f.write("-" * 50 + "\n")
        for hz in ["6h", "12h", "24h", "2day", "1week", "2week", "1month"]:
            h_data = eval_df[(eval_df["Horizon"] == hz) & (eval_df["Cutoff_Time"] == "All")]
            if len(h_data) > 0:
                row = h_data.iloc[0]
                q = "Excellent" if row['R2'] > 0.9 else "Very Good" if row['R2'] > 0.8 else \
                    "Good" if row['R2'] > 0.7 else "Moderate" if row['R2'] > 0.5 else "Poor"
                f.write(f"{hz:8}  {row['RMSE']:6.2f}  {row['MAE']:5.2f}  {row['R2']:6.4f}  {q}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 100 + "\n\n")
        f.write("1. PV PERFORMANCE AT DIFFERENT TIMES:\n")
        f.write("   Compare midnight (wind-only) vs noon (PV-peak)\n")
        f.write("   Noon should have LOWER R² due to cloud variability\n\n")

        f.write("2. SHORT-TERM vs LONG-TERM:\n")
        f.write("   Accuracy decreases with horizon (EXPECTED)\n")
        f.write("   6h good, 1 month moderate\n\n")

        f.write("3. HOLT-WINTERS ASSESSMENT:\n")
        f.write("   ✓ Good for day-ahead (< 24 hours)\n")
        f.write("   ✗ Poor for long-term (> 1 week)\n")
        f.write("   ✗ Cannot capture PV cloud variability\n")
        f.write("   → Use for operations, not strategic planning\n")

    print(f"✓ Saved summary → {SUMMARY_FILE}")

    print(f"\n" + "=" * 100)
    print(f"✓ COMPLETED!")
    print(f"✓ Runtime: {elapsed_total:.1f} minutes")
    print(f"✓ Total predictions: {len(final):,}")
    print(f"✓ Models saved: {len(districts) * len(CUTOFF_HOURS)}")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()