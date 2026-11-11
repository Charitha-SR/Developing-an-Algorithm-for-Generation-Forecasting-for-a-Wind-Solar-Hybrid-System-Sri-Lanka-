"""
GRU-Final Live Demo Script with GUI (array-based, self-contained)
============================================================
Folder structure expected (relative to this script, in src/):

    Final Model/
    |-- artifacts/
    |   |-- scaling_stats.json
    |   |-- category_codes.json
    |   |-- features_test.npy
    |   |-- targets_test.npy
    |   |-- test_district_codes.npy
    |   |-- test_solar_codes.npy
    |   |-- test_wind_codes.npy
    |   `-- test_is_context.npy
    |-- Model/
    |   `-- gru_best_model_v2.pt
    |-- src/
    |   `-- (this script)
    `-- requirements.txt

No parquet file or climatology_lookup.csv is needed here - features_test.npy
already contains every feature (including climatology) pre-built, in the
exact order the model was trained on.

DATE HANDLING NOTE:
features_test.npy / targets_test.npy have no stored calendar dates - they
are raw row arrays. This script reconstructs dates using two facts already
verified in the project's own dataset: (1) the data is a complete, gap-free
hourly series with zero missing values, and (2) the test period is exactly
2023-01-01 to 2025-12-31. So the first non-context row in each district/
system-size group is assumed to be 2023-01-01 00:00, with every following
row exactly +1 hour. A sanity check below verifies this assumption holds
for the selected group before trusting it.
============================================================
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ======================= CONFIG: FILE PATHS =======================
# Paths are relative to this script's own location, so the demo works
# from this folder structure on ANY computer.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Final Model/src
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "..", "Model")
MODEL_PATH = os.path.join(MODEL_DIR, "gru_best_model_v2.pt")

LOOKBACK_HOURS = 336
HIDDEN_SIZE = 128
EMBED_DIM_DISTRICT = 8
EMBED_DIM_SIZE = 4
N_HORIZONS = 8

HORIZON_ORDER = ["1h", "6h", "12h", "24h", "2d", "1week", "2week", "1month"]
HORIZON_STEPS = {"1h": 1, "6h": 6, "12h": 12, "24h": 24, "2d": 48,
                 "1week": 168, "2week": 336, "1month": 720}

TEST_PERIOD_START = datetime(2023, 1, 1, 0, 0)  # verified test-set start date

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------
# MODEL DEFINITION (must match GRU-Final training script exactly)
# ---------------------------------------------------------------
class GRUForecastNet(nn.Module):
    def __init__(self, n_features, n_districts, n_solar_sizes, n_wind_sizes,
                 hidden_size, embed_dim_district, embed_dim_size, n_horizons):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size,
                           num_layers=2, batch_first=True, dropout=0.1)
        self.district_emb = nn.Embedding(n_districts, embed_dim_district)
        self.solar_emb = nn.Embedding(n_solar_sizes, embed_dim_size)
        self.wind_emb = nn.Embedding(n_wind_sizes, embed_dim_size)

        combined_dim = hidden_size + embed_dim_district + 2 * embed_dim_size
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, 128), nn.ReLU(), nn.Dropout(0.1),
        )
        self.solar_head = nn.Linear(128, n_horizons)
        self.wind_head = nn.Linear(128, n_horizons)
        self.total_head = nn.Linear(128, n_horizons)

    def forward(self, window, district_code, solar_code, wind_code):
        _, h_n = self.gru(window)
        gru_summary = h_n[-1]
        d_emb = self.district_emb(district_code)
        s_emb = self.solar_emb(solar_code)
        w_emb = self.wind_emb(wind_code)
        combined = torch.cat([gru_summary, d_emb, s_emb, w_emb], dim=1)
        shared_out = self.shared(combined)
        return torch.cat([self.solar_head(shared_out),
                           self.wind_head(shared_out),
                           self.total_head(shared_out)], dim=1)


# ---------------------------------------------------------------
# ARTIFACT LOADING (done once, cached for repeated GUI runs)
# ---------------------------------------------------------------
_cache = {}

def load_artifacts():
    if _cache:
        return _cache

    with open(os.path.join(ARTIFACTS_DIR, "scaling_stats.json")) as f:
        scaling = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "category_codes.json")) as f:
        cat_codes = json.load(f)

    features = np.load(os.path.join(ARTIFACTS_DIR, "features_test.npy"))
    targets = np.load(os.path.join(ARTIFACTS_DIR, "targets_test.npy"))
    d_codes = np.load(os.path.join(ARTIFACTS_DIR, "test_district_codes.npy"))
    s_codes = np.load(os.path.join(ARTIFACTS_DIR, "test_solar_codes.npy"))
    w_codes = np.load(os.path.join(ARTIFACTS_DIR, "test_wind_codes.npy"))
    is_context = np.load(os.path.join(ARTIFACTS_DIR, "test_is_context.npy"))

    _cache.update(dict(
        feat_mean=np.array(scaling["feature_mean"], dtype=np.float32),
        feat_std=np.array(scaling["feature_std"], dtype=np.float32),
        target_mean=np.array(scaling["target_mean"], dtype=np.float32),
        target_std=np.array(scaling["target_std"], dtype=np.float32),
        cat_codes=cat_codes,
        features=features, targets=targets,
        d_codes=d_codes, s_codes=s_codes, w_codes=w_codes,
        is_context=is_context,
        n_districts=len(cat_codes["district_code"]),
        n_solar_sizes=len(cat_codes["solar_code"]),
        n_wind_sizes=len(cat_codes["wind_code"]),
        n_features=len(scaling["feature_mean"]),
    ))
    return _cache


def find_group_bounds(d_codes, s_codes, w_codes, district_code, solar_code, wind_code):
    """Find the contiguous row range for one district/solar/wind combination."""
    mask = (d_codes == district_code) & (s_codes == solar_code) & (w_codes == wind_code)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError("No test data found for this district/solar/wind combination.")
    if idx[-1] - idx[0] + 1 != len(idx):
        raise ValueError("Selected combination's rows are not contiguous - unexpected data layout.")
    return idx[0], idx[-1]  # inclusive bounds


def anchor_row_for_test_start(group_start, group_end, is_context):
    """First non-context row in the group = 2023-01-01 00:00 (verified assumption)."""
    for t in range(group_start, group_end + 1):
        if not is_context[t]:
            return t
    raise ValueError("No non-context (valid test) rows found in this group.")


def datetime_to_row(anchor_row, target_dt):
    hours_offset = int((target_dt - TEST_PERIOD_START).total_seconds() // 3600)
    return anchor_row + hours_offset


# ---------------------------------------------------------------
# FORECASTING LOGIC
# ---------------------------------------------------------------
def execute_forecast(district, year, month, day, hour, solar_size, wind_size, horizon):
    try:
        art = load_artifacts()
        cat_codes = art["cat_codes"]

        district_code = cat_codes["district_code"][district]
        solar_code = cat_codes["solar_code"][str(solar_size)]
        wind_code = cat_codes["wind_code"][str(wind_size)]

        group_start, group_end = find_group_bounds(
            art["d_codes"], art["s_codes"], art["w_codes"],
            district_code, solar_code, wind_code
        )
        anchor_row = anchor_row_for_test_start(group_start, group_end, art["is_context"])

        # Sanity check: anchor row's group should span roughly 3 years (2023-2025)
        expected_hours = 3 * 8760 + 24  # 2024 is a leap year (+24h), rough check
        actual_span = group_end - anchor_row + 1
        if abs(actual_span - expected_hours) > 48:
            messagebox.showwarning(
                "Date Assumption Warning",
                f"The reconstructed test period length ({actual_span} hours) does not "
                f"closely match the expected ~2023-2025 span. Displayed dates may be "
                f"incorrect for this combination - verify before trusting results."
            )

        target_dt = datetime(year, month, day, hour)
        t = datetime_to_row(anchor_row, target_dt)

        if t < group_start + LOOKBACK_HOURS - 1 or t > group_end:
            messagebox.showerror(
                "Data Error",
                "This date is out of range for a 336-hour lookback within the test period "
                "(2023-01-01 onward), or falls outside the available test data for this "
                "district/system-size combination."
            )
            return

        feat_mean, feat_std = art["feat_mean"], art["feat_std"]
        target_mean, target_std = art["target_mean"], art["target_std"]

        window = art["features"][t - LOOKBACK_HOURS + 1: t + 1].astype(np.float32)
        window_norm = (window - feat_mean) / feat_std
        window_tensor = torch.tensor(window_norm).unsqueeze(0).to(DEVICE)

        d_tensor = torch.tensor([district_code], dtype=torch.long).to(DEVICE)
        s_tensor = torch.tensor([solar_code], dtype=torch.long).to(DEVICE)
        w_tensor = torch.tensor([wind_code], dtype=torch.long).to(DEVICE)

        model = GRUForecastNet(art["n_features"], art["n_districts"], art["n_solar_sizes"],
                                art["n_wind_sizes"], HIDDEN_SIZE, EMBED_DIM_DISTRICT,
                                EMBED_DIM_SIZE, N_HORIZONS).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.eval()

        with torch.no_grad():
            pred_norm = model(window_tensor, d_tensor, s_tensor, w_tensor).cpu().numpy()[0]
        pred = pred_norm * target_std + target_mean  # length 24: [Solar x8, Wind x8, Total x8]

        h_idx = HORIZON_ORDER.index(horizon)
        pred_solar = max(pred[h_idx], 0.0)
        pred_wind = max(pred[8 + h_idx], 0.0)
        pred_total = max(pred[16 + h_idx], 0.0)

        # Actual value: targets_test.npy is already un-normalised (raw kW)
        actual_vec = art["targets"][t]
        actual_solar = actual_vec[h_idx]
        actual_wind = actual_vec[8 + h_idx]
        actual_total = actual_vec[16 + h_idx]

        actual_available = not (np.isnan(actual_solar) or np.isnan(actual_wind) or np.isnan(actual_total))

        print("\n" + "=" * 60)
        print(f"Forecast: {district} | {target_dt} | Solar={solar_size}kW Wind={wind_size}kW | {horizon}")
        print("=" * 60)
        if actual_available:
            print(f"{'Power Type':<12}{'Predicted (kW)':<18}{'Actual (kW)':<15}")
            print(f"{'Solar':<12}{pred_solar:<18.2f}{actual_solar:<15.2f}")
            print(f"{'Wind':<12}{pred_wind:<18.2f}{actual_wind:<15.2f}")
            print(f"{'Total':<12}{pred_total:<18.2f}{actual_total:<15.2f}")
        else:
            print(f"{'Power Type':<12}{'Predicted (kW)':<18}")
            print(f"{'Solar':<12}{pred_solar:<18.2f}")
            print(f"{'Wind':<12}{pred_wind:<18.2f}")
            print(f"{'Total':<12}{pred_total:<18.2f}")
            print("\n(Actual value not available - too close to the end of the test period "
                  "for this horizon.)")

        # ---- Plot ----
        labels = ["Solar", "Wind", "Total"]
        predicted = [pred_solar, pred_wind, pred_total]
        x = np.arange(len(labels))

        plt.figure(figsize=(6, 4))
        if actual_available:
            actual = [actual_solar, actual_wind, actual_total]
            plt.bar(x - 0.15, predicted, width=0.3, label="Predicted")
            plt.bar(x + 0.15, actual, width=0.3, label="Actual")
        else:
            plt.bar(x, predicted, width=0.3, label="Predicted")
        plt.xticks(x, labels)
        plt.ylabel("Power (kW)")
        plt.title(f"GRU-Final: Predicted vs Actual\n{district}, {target_dt} + {horizon}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Execution Error", f"An error occurred:\n{str(e)}")


# ---------------------------------------------------------------
# GUI APPLICATION
# ---------------------------------------------------------------
def run_gui():
    root = tk.Tk()
    root.title("GRU-Final Forecasting Demo")
    root.geometry("380x350")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use('clam')

    main_frame = ttk.Frame(root, padding=(20, 10))
    main_frame.pack(fill='both', expand=True)

    districts = ["Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya", "Galle", "Matara",
                 "Hambantota", "Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Mullaitivu", "Batticaloa",
                 "Ampara", "Trincomalee", "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa",
                 "Badulla", "Moneragala", "Ratnapura", "Kegalle"]

    ttk.Label(main_frame, text="District:").grid(row=0, column=0, sticky="w", pady=5)
    cb_dist = ttk.Combobox(main_frame, values=districts, state="readonly", width=15)
    cb_dist.set("Colombo")
    cb_dist.grid(row=0, column=1, sticky="w", pady=5)

    ttk.Label(main_frame, text="Date (Y/M/D):").grid(row=1, column=0, sticky="w", pady=5)
    date_frame = ttk.Frame(main_frame)
    date_frame.grid(row=1, column=1, sticky="w", pady=5)
    cb_year = ttk.Combobox(date_frame, values=["2023"], state="readonly", width=5)
    cb_year.set("2023")
    cb_year.pack(side="left", padx=(0, 2))
    cb_month = ttk.Combobox(date_frame, values=[str(i) for i in range(1, 7)], state="readonly", width=3)
    cb_month.set("1")
    cb_month.pack(side="left", padx=2)
    cb_day = ttk.Combobox(date_frame, values=[str(i) for i in range(1, 32)], state="readonly", width=3)
    cb_day.set("17")
    cb_day.pack(side="left", padx=2)

    ttk.Label(main_frame, text="Hour (0-23):").grid(row=2, column=0, sticky="w", pady=5)
    cb_hour = ttk.Combobox(main_frame, values=[str(i) for i in range(24)], state="readonly", width=15)
    cb_hour.set("0")
    cb_hour.grid(row=2, column=1, sticky="w", pady=5)

    ttk.Label(main_frame, text="Solar Size (kW):").grid(row=3, column=0, sticky="w", pady=5)
    cb_solar = ttk.Combobox(main_frame, values=["1", "5", "100"], state="readonly", width=15)
    cb_solar.set("100")
    cb_solar.grid(row=3, column=1, sticky="w", pady=5)

    ttk.Label(main_frame, text="Wind Size (kW):").grid(row=4, column=0, sticky="w", pady=5)
    cb_wind = ttk.Combobox(main_frame, values=["10", "50", "100"], state="readonly", width=15)
    cb_wind.set("100")
    cb_wind.grid(row=4, column=1, sticky="w", pady=5)

    ttk.Label(main_frame, text="Horizon:").grid(row=5, column=0, sticky="w", pady=5)
    cb_horizon = ttk.Combobox(main_frame, values=HORIZON_ORDER, state="readonly", width=15)
    cb_horizon.set("1month")
    cb_horizon.grid(row=5, column=1, sticky="w", pady=5)

    def on_run():
        execute_forecast(
            cb_dist.get(), int(cb_year.get()), int(cb_month.get()), int(cb_day.get()),
            int(cb_hour.get()), int(cb_solar.get()), int(cb_wind.get()), cb_horizon.get()
        )

    btn_run = ttk.Button(main_frame, text="Run Forecast", command=on_run)
    btn_run.grid(row=6, column=0, columnspan=2, pady=20)

    root.mainloop()


if __name__ == "__main__":
    run_gui()