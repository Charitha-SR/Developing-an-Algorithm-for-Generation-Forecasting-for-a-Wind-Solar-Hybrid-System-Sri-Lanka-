

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats as scipy_stats
import os
import time
import json
from datetime import datetime, timedelta


# ========================== CONFIG ==========================

INPUT_DIR  = "data"
OUTPUT_DIR = "results/cnn_lstm"

TRAIN_FILE = os.path.join(INPUT_DIR, "train_dataset.parquet")
TEST_FILE  = os.path.join(INPUT_DIR, "test_dataset.parquet")

FOLDS = [
    {"name": "Fold_1", "train_end": 2012, "val_year": 2013,
     "test_start": 2014, "test_end": 2016},
    {"name": "Fold_2", "train_end": 2015, "val_year": 2016,
     "test_start": 2017, "test_end": 2019},
    {"name": "Fold_3", "train_end": 2018, "val_year": 2019,
     "test_start": 2020, "test_end": 2022},
    {"name": "Fold_4", "train_end": 2021, "val_year": 2022,
     "test_start": 2023, "test_end": 2025},
]

HORIZON_NAMES = ["1h", "6h", "12h", "24h", "2d", "1week", "2week", "1month"]
HORIZON_STEPS = [1, 6, 12, 24, 48, 168, 336, 720]

LOOKBACK    = 336
MAX_HORIZON = 720

# === SPEED SETTINGS ===
CNN_CHANNELS     = [64, 128, 64]
CNN_KERNELS      = [7, 5, 3]
LSTM_HIDDEN      = 128          # reduced from 256
LSTM_LAYERS      = 2
DROPOUT          = 0.1
EMBED_DIM        = 8
BATCH_SIZE       = 1024         # Colab T4 can handle more
STEPS_PER_EPOCH  = 4000
EPOCHS           = 30
LR               = 1e-3
WEIGHT_DECAY     = 1e-4
WARMUP_STEPS     = 500
PATIENCE         = 7
GRAD_CLIP        = 1.0
NUM_WORKERS      = 2            # Linux (Colab) supports workers

SKIP_COMPLETED   = True
# ============================================================


def fmt_time(s):
    if s < 60:     return f"{s:.0f}s"
    elif s < 3600: return f"{s / 60:.1f}min"
    else:          return f"{int(s // 3600)}h {int((s % 3600) // 60)}min"


# ----------------------------------------------------------------
# DATA
# ----------------------------------------------------------------
SEQ_FEATURES = [
    "GHI", "Wind_Speed",
    "Solar_Power_Output_KW", "Wind_Power_Output_KW",
    "Total_Hybrid_Power_Output_KW",
    "Hour_sin", "Hour_cos",
    "Month_sin", "Month_cos",
    "DayOfYear_sin", "DayOfYear_cos",
]
N_FEATURES = len(SEQ_FEATURES)


def prepare_full_data():
    print("  Loading parquet files...")
    train_df = pd.read_parquet(TRAIN_FILE)
    test_df  = pd.read_parquet(TEST_FILE)
    print(f"    Train: {len(train_df):>10,} rows")
    print(f"    Test:  {len(test_df):>10,} rows")

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    del train_df, test_df

    full_df = full_df.sort_values(["District", "Datetime"]).reset_index(drop=True)

    full_df["Hour_sin"]      = np.sin(2 * np.pi * full_df["Hour"] / 24)
    full_df["Hour_cos"]      = np.cos(2 * np.pi * full_df["Hour"] / 24)
    full_df["Month_sin"]     = np.sin(2 * np.pi * (full_df["Month"] - 1) / 12)
    full_df["Month_cos"]     = np.cos(2 * np.pi * (full_df["Month"] - 1) / 12)
    full_df["DayOfYear"]     = full_df["Datetime"].dt.dayofyear
    full_df["DayOfYear_sin"] = np.sin(2 * np.pi * full_df["DayOfYear"] / 365)
    full_df["DayOfYear_cos"] = np.cos(2 * np.pi * full_df["DayOfYear"] / 365)

    print(f"    Combined: {len(full_df):>10,} rows")
    return full_df


def build_district_arrays(full_df):
    districts = sorted(full_df["District"].unique())
    district_to_id = {d: i for i, d in enumerate(districts)}
    arrays = []
    for d_name in districts:
        d_df = full_df[full_df["District"] == d_name].sort_values("Datetime")
        arrays.append({
            "features":    d_df[SEQ_FEATURES].values.astype(np.float32),
            "power":       d_df["Total_Hybrid_Power_Output_KW"].values.astype(np.float32),
            "year":        d_df["Year"].values.astype(np.int32),
            "datetime":    d_df["Datetime"].values,
            "district_id": district_to_id[d_name],
            "name":        d_name,
        })
    return arrays, districts, district_to_id


def build_indices(district_arrays, year_filter_fn):
    indices = []
    for d_idx, d in enumerate(district_arrays):
        n = len(d["power"])
        for t in range(LOOKBACK, n - MAX_HORIZON):
            if year_filter_fn(d["year"][t]):
                indices.append((d_idx, t))
    return indices


def compute_normalization(district_arrays, year_filter_fn):
    all_f, all_p = [], []
    for d in district_arrays:
        mask = np.array([year_filter_fn(y) for y in d["year"]], dtype=bool)
        all_f.append(d["features"][mask])
        all_p.append(d["power"][mask])
    features = np.concatenate(all_f, axis=0)
    power = np.concatenate(all_p, axis=0)
    fm, fs = features.mean(axis=0), features.std(axis=0)
    fs[fs < 1e-8] = 1.0
    return fm, fs, power.mean(), power.std()


# ----------------------------------------------------------------
# COVERAGE SAMPLER
# ----------------------------------------------------------------
class CoverageSampler(Sampler):
    def __init__(self, n_total, samples_per_epoch, seed=42):
        self.n_total = n_total
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)
        self.permutation = self.rng.permutation(n_total)
        self.cursor = 0

    def __iter__(self):
        chunks = []
        remaining = self.samples_per_epoch
        while remaining > 0:
            available = len(self.permutation) - self.cursor
            take = min(remaining, available)
            chunks.append(self.permutation[self.cursor: self.cursor + take])
            self.cursor += take
            remaining -= take
            if self.cursor >= len(self.permutation):
                self.permutation = self.rng.permutation(self.n_total)
                self.cursor = 0
        return iter(np.concatenate(chunks).tolist())

    def __len__(self):
        return self.samples_per_epoch


# ----------------------------------------------------------------
# DATASET
# ----------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, district_arrays, indices, feat_mean, feat_std,
                 power_mean, power_std):
        self.data = district_arrays
        self.indices = indices
        self.fm, self.fs = feat_mean, feat_std
        self.pm, self.ps = power_mean, power_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        d_idx, t = self.indices[idx]
        d = self.data[d_idx]
        x = d["features"][t - LOOKBACK:t].copy()
        x = (x - self.fm) / self.fs
        targets = np.array([d["power"][t + h] for h in HORIZON_STEPS],
                           dtype=np.float32)
        targets_norm = (targets - self.pm) / self.ps
        return (torch.from_numpy(x),
                torch.tensor(d["district_id"], dtype=torch.long),
                torch.from_numpy(targets_norm),
                torch.from_numpy(targets))


# ----------------------------------------------------------------
# CNN-LSTM MODEL
# ----------------------------------------------------------------
class CNNLSTMForecaster(nn.Module):
    def __init__(self, n_features, n_districts, cnn_channels, cnn_kernels,
                 lstm_hidden, lstm_layers, dropout, embed_dim, n_horizons):
        super().__init__()
        self.district_emb = nn.Embedding(n_districts, embed_dim)

        self.conv_layers = nn.ModuleList()
        self.bn_layers   = nn.ModuleList()
        in_ch = n_features
        for out_ch, kernel in zip(cnn_channels, cnn_kernels):
            self.conv_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2))
            self.bn_layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        head_input = lstm_hidden + embed_dim
        self.shared = nn.Sequential(
            nn.Linear(head_input, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(n_horizons)
        ])

    def forward(self, x_seq, district_idx):
        x = x_seq.transpose(1, 2)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn(conv(x)))
        x = self.pool(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        last_hidden = self.dropout(lstm_out[:, -1, :])

        d_emb = self.district_emb(district_idx)
        combined = torch.cat([last_hidden, d_emb], dim=1)
        shared = self.shared(combined)
        return torch.cat([head(shared) for head in self.heads], dim=1)


# ----------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, global_step):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, dist, targets_norm, _ in loader:
        x = x.to(device)
        dist = dist.to(device)
        targets_norm = targets_norm.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=True):
            preds = model(x, dist)
            loss = nn.functional.mse_loss(preds, targets_norm)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        if global_step < WARMUP_STEPS:
            warmup_lr = LR * (global_step + 1) / WARMUP_STEPS
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr

        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), global_step


def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, dist, targets_norm, _ in loader:
            x = x.to(device)
            dist = dist.to(device)
            targets_norm = targets_norm.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                preds = model(x, dist)
                loss = nn.functional.mse_loss(preds, targets_norm)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_on_test(model, loader, power_mean, power_std, device):
    model.eval()
    all_p, all_t, all_d = [], [], []
    with torch.no_grad():
        for x, dist, _, targets_raw in loader:
            x, dist = x.to(device), dist.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                pn = model(x, dist)
            all_p.append(pn.float().cpu().numpy() * power_std + power_mean)
            all_t.append(targets_raw.numpy())
            all_d.append(dist.cpu().numpy())
    return np.concatenate(all_p), np.concatenate(all_t), np.concatenate(all_d)


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
        "RMSE": round(rmse, 4), "MAE": round(mae, 4),
        "MAPE_%": round(mape, 2), "R2": round(r2, 6),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("  CNN-LSTM - FAST CROSS-VALIDATION (4 FOLDS) - COLAB")
    print(f"  CNN: {CNN_CHANNELS} | LSTM: h={LSTM_HIDDEN}, L={LSTM_LAYERS}")
    print(f"  Batch: {BATCH_SIZE} | Steps/epoch: {STEPS_PER_EPOCH}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n  [STEP 1] Loading data...")
    full_df = prepare_full_data()

    print("\n  [STEP 2] Building per-district arrays...")
    district_arrays, districts, district_to_id = build_district_arrays(full_df)
    n_districts = len(districts)
    print(f"    {n_districts} districts, {N_FEATURES} features")
    del full_df

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_fold_metrics = []

    for fold_idx, fold in enumerate(FOLDS):
        fold_name = fold["name"]
        fold_dir  = os.path.join(OUTPUT_DIR, fold_name)
        model_dir = os.path.join(fold_dir, "models")

        eval_path = os.path.join(fold_dir, f"{fold_name}_evaluation.csv")
        if SKIP_COMPLETED and os.path.exists(eval_path):
            print(f"\n  SKIP  {fold_name} - already completed")
            try:
                existing = pd.read_csv(eval_path)
                all_fold_metrics.extend(existing.to_dict("records"))
            except Exception:
                pass
            continue

        os.makedirs(model_dir, exist_ok=True)
        fold_start = time.time()

        print(f"\n  {'=' * 66}")
        print(f"  {fold_name}:  Train 1987-{fold['train_end']}  |  "
              f"Val {fold['val_year']}  |  "
              f"Test {fold['test_start']}-{fold['test_end']}")
        print(f"  {'=' * 66}")

        print("    Building indices...", end=" ", flush=True)
        t0 = time.time()
        train_idx = build_indices(district_arrays,
                                  lambda y: y <= fold["train_end"])
        val_idx   = build_indices(district_arrays,
                                  lambda y: y == fold["val_year"])
        test_idx  = build_indices(district_arrays,
                                  lambda y: fold["test_start"] <= y <= fold["test_end"])
        print(f"done in {fmt_time(time.time() - t0)}")
        print(f"    Train: {len(train_idx):>10,}  |  "
              f"Val: {len(val_idx):>10,}  |  "
              f"Test: {len(test_idx):>10,}")

        print("    Normalization...", end=" ", flush=True)
        feat_mean, feat_std, power_mean, power_std = compute_normalization(
            district_arrays, lambda y: y <= fold["train_end"])
        print("done")

        train_ds = SequenceDataset(district_arrays, train_idx,
                                   feat_mean, feat_std, power_mean, power_std)
        val_ds   = SequenceDataset(district_arrays, val_idx,
                                   feat_mean, feat_std, power_mean, power_std)
        test_ds  = SequenceDataset(district_arrays, test_idx,
                                   feat_mean, feat_std, power_mean, power_std)

        train_sampler = CoverageSampler(len(train_ds),
                                        BATCH_SIZE * STEPS_PER_EPOCH)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=train_sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2,
                                  shuffle=False, num_workers=NUM_WORKERS,
                                  pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE * 2,
                                  shuffle=False, num_workers=NUM_WORKERS,
                                  pin_memory=True)

        model = CNNLSTMForecaster(
            n_features=N_FEATURES, n_districts=n_districts,
            cnn_channels=CNN_CHANNELS, cnn_kernels=CNN_KERNELS,
            lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
            dropout=DROPOUT, embed_dim=EMBED_DIM,
            n_horizons=len(HORIZON_STEPS),
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"    Model params: {total_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                       weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3)
        scaler = torch.amp.GradScaler()

        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0
        best_path = os.path.join(model_dir, f"cnn_lstm_{fold_name}_best.pt")
        training_log = []

        print(f"\n    {'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} "
              f"{'LR':>12} {'Time':>8} {'Status':>10}")
        print(f"    {'_' * 62}")

        for epoch in range(1, EPOCHS + 1):
            ep_start = time.time()

            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, scaler, device, global_step)
            val_loss = validate_one_epoch(model, val_loader, device)

            if global_step >= WARMUP_STEPS:
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            ep_time = time.time() - ep_start

            status = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
                status = "BEST"
            else:
                patience_counter += 1
                status = "STOP" if patience_counter >= PATIENCE else f"wait {patience_counter}/{PATIENCE}"

            print(f"    {epoch:>5} {train_loss:>12.6f} {val_loss:>12.6f} "
                  f"{current_lr:>12.8f} {fmt_time(ep_time):>8} {status:>10}")

            training_log.append({
                "Epoch": epoch, "Train_Loss": round(train_loss, 6),
                "Val_Loss": round(val_loss, 6),
                "LR": round(current_lr, 8), "Time_s": round(ep_time, 1),
            })

            if status == "STOP":
                print(f"\n    Early stopping at epoch {epoch}")
                break

        pd.DataFrame(training_log).to_csv(
            os.path.join(fold_dir, "training_log.csv"), index=False)

        model.load_state_dict(torch.load(best_path, weights_only=True))
        print(f"\n    Best model loaded (val_loss={best_val_loss:.6f})")

        with open(os.path.join(model_dir, "norm_stats.json"), "w") as f:
            json.dump({"feat_mean": feat_mean.tolist(),
                       "feat_std": feat_std.tolist(),
                       "power_mean": float(power_mean),
                       "power_std": float(power_std)}, f, indent=2)

        print("    Evaluating...", end=" ", flush=True)
        t0 = time.time()
        all_preds, all_targets, all_dist_ids = evaluate_on_test(
            model, test_loader, power_mean, power_std, device)
        print(f"done in {fmt_time(time.time() - t0)}")

        id_to_district = {v: k for k, v in district_to_id.items()}
        dist_names = np.array([id_to_district[d] for d in all_dist_ids])

        all_datetimes = []
        for d_idx, t in test_idx:
            all_datetimes.append(district_arrays[d_idx]["datetime"][t])
        all_datetimes = np.array(all_datetimes)

        fold_metrics = []
        fold_predictions = {}

        for h_i, h_name in enumerate(HORIZON_NAMES):
            actual    = all_targets[:, h_i]
            predicted = np.maximum(all_preds[:, h_i], 0.0)

            metrics = compute_metrics(actual, predicted)
            metrics["Fold"]       = fold_name
            metrics["Horizon"]    = h_name
            metrics["Train_Seqs"] = len(train_idx)
            metrics["Test_Seqs"]  = len(test_idx)
            fold_metrics.append(metrics)

            print(f"    {h_name:<8}  RMSE: {metrics['RMSE']:.4f}  |  "
                  f"MAE: {metrics['MAE']:.4f}  |  "
                  f"MAPE: {metrics['MAPE_%']:.2f}%  |  "
                  f"R2: {metrics['R2']:.6f}")

            for district in districts:
                d_mask = dist_names == district
                if d_mask.sum() == 0:
                    continue
                d_result = pd.DataFrame({
                    "Datetime": all_datetimes[d_mask], "Horizon": h_name,
                    "Actual": actual[d_mask], "Predicted": predicted[d_mask],
                    "Residual": predicted[d_mask] - actual[d_mask],
                })
                if district not in fold_predictions:
                    fold_predictions[district] = []
                fold_predictions[district].append(d_result)

            d_rows = []
            for district in sorted(districts):
                d_mask = dist_names == district
                if d_mask.sum() == 0:
                    continue
                d_m = compute_metrics(actual[d_mask], predicted[d_mask])
                d_m["District"] = district
                d_m["Horizon"]  = h_name
                d_rows.append(d_m)
            pd.DataFrame(d_rows).to_csv(
                os.path.join(fold_dir, f"district_metrics_{h_name}.csv"),
                index=False)

        print(f"\n    Saving Excel...", end=" ", flush=True)
        excel_path = os.path.join(fold_dir, f"{fold_name}_predictions.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for district in sorted(fold_predictions.keys()):
                d_all = pd.concat(fold_predictions[district],
                                  ignore_index=True)
                d_all = d_all.sort_values(["Datetime", "Horizon"])
                d_all.to_excel(writer, sheet_name=district[:31],
                               index=False)
        print("done")

        fold_df = pd.DataFrame(fold_metrics)
        fold_df.to_csv(eval_path, index=False)

        res_parts = []
        for d_list in fold_predictions.values():
            for d_df in d_list:
                res_parts.append(d_df[["Horizon", "Residual"]])
        pd.concat(res_parts, ignore_index=True).to_csv(
            os.path.join(fold_dir, f"{fold_name}_residuals.csv"),
            index=False)

        all_fold_metrics.extend(fold_metrics)

        del model, optimizer, scheduler, scaler
        del train_ds, val_ds, test_ds
        del train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

        fold_time = time.time() - fold_start
        print(f"\n    >>> {fold_name} completed in {fmt_time(fold_time)}")

    # ---- SUMMARY ----
    print(f"\n{'=' * 70}")
    print("  CNN-LSTM CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 70}")

    summary_df = pd.DataFrame(all_fold_metrics)
    summary_path = os.path.join(OUTPUT_DIR, "CV_summary_all_folds.csv")
    summary_df.to_csv(summary_path, index=False)

    for h_name in HORIZON_NAMES:
        h_data = summary_df[summary_df["Horizon"] == h_name]
        if len(h_data) == 0:
            continue
        print(f"\n  {h_name:<10}  "
              f"RMSE: {h_data['RMSE'].mean():.4f} +/- {h_data['RMSE'].std():.4f}  |  "
              f"R2: {h_data['R2'].mean():.6f} +/- {h_data['R2'].std():.6f}")

    print(f"\n  Summary -> {summary_path}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  TOTAL TIME: {fmt_time(total_time)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()