
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import json
import os
import time

# ======================= CONFIG =======================
# Reminder: after running Stage 2 v3, copy its output files from
# D:\...\DL_Output_File into C:\DL_Input_data (overwriting the old
# 168-hour versions) if you're keeping the fast-NVMe-read setup from
# before - otherwise this script will still be reading the OLD arrays.
INPUT_DIR = "data/dl_features"
OUTPUT_DIR = "data/dl_output"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "gru_checkpoint_v2.pt")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "gru_best_model_v2.pt")

BATCH_SIZE = 256
STEPS_PER_EPOCH = 4000        # doubled from 2000 - see module docstring for the checkpoint-granularity reasoning
NUM_EPOCHS = 500              # high safety ceiling only - TRAIN_TIME_BUDGET_HOURS below is the REAL stopping condition
TRAIN_TIME_BUDGET_HOURS = 9.0 # stops cleanly (with a valid saved checkpoint) once this much wall-clock time has
                               # elapsed SINCE THIS SCRIPT STARTED - applies per run, so if you resume tomorrow
                               # night, you get another full ~9-hour budget from that new start time
LEARNING_RATE = 1e-3
WARMUP_STEPS = 500            # number of successful weight updates over which LR ramps up from ~0 to LEARNING_RATE -
                               # see explanation below for why this matters for early-training stability
HIDDEN_SIZE = 128
EMBED_DIM_DISTRICT = 8
EMBED_DIM_SIZE = 4
N_HORIZONS = 8
NUM_WORKERS = 0               # keep at 0 on Windows - see earlier fix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# ========================================================

with open(os.path.join(INPUT_DIR, "scaling_stats.json")) as f:
    SCALING = json.load(f)
with open(os.path.join(INPUT_DIR, "category_codes.json")) as f:
    CATEGORY_CODES = json.load(f)
with open(os.path.join(INPUT_DIR, "group_metadata_train.json")) as f:
    TRAIN_META = json.load(f)

N_FEATURES = TRAIN_META["n_features"]
N_TARGETS = TRAIN_META["n_targets"]
LOOKBACK_HOURS = TRAIN_META["lookback_hours"]   # now read from metadata, not hardcoded
N_DISTRICTS = len(CATEGORY_CODES["district_code"])
N_SOLAR_SIZES = len(CATEGORY_CODES["solar_code"])
N_WIND_SIZES = len(CATEGORY_CODES["wind_code"])

print(f"Loaded config from Stage 2: {N_FEATURES} features, lookback = {LOOKBACK_HOURS} hours")

FEATURE_MEAN = np.array(SCALING["feature_mean"], dtype=np.float32)
FEATURE_STD = np.array(SCALING["feature_std"], dtype=np.float32)
TARGET_MEAN = np.array(SCALING["target_mean"], dtype=np.float32)
TARGET_STD = np.array(SCALING["target_std"], dtype=np.float32)

# ---- Per-horizon loss weights ----
# Order must match Stage 2's TARGET_COLS: for each output (Solar, Wind,
# Total), the 8 horizons in this exact order: 1h,6h,12h,24h,2d,1week,2week,1month
HORIZON_ORDER = ["1h", "6h", "12h", "24h", "2d", "1week", "2week", "1month"]
HORIZON_WEIGHT_MAP = {
    "1h": 1.0, "6h": 1.0, "12h": 1.0, "24h": 1.2,
    "2d": 1.3, "1week": 1.5, "2week": 1.7, "1month": 2.0,
}
_single_block = np.array([HORIZON_WEIGHT_MAP[h] for h in HORIZON_ORDER], dtype=np.float32)
TARGET_WEIGHTS = np.tile(_single_block, 3)  # repeat for Solar, Wind, Total blocks -> length 24
TARGET_WEIGHTS_TENSOR = torch.tensor(TARGET_WEIGHTS, dtype=torch.float32, device=DEVICE)


def build_valid_windows_train(group_meta_list, lookback):
    t_list, d_list, s_list, w_list = [], [], [], []
    for g in group_meta_list:
        start, end = g["start"], g["end"]
        first_valid = start + lookback - 1
        if first_valid >= end:
            continue
        ts = np.arange(first_valid, end, dtype=np.int64)
        t_list.append(ts)
        d_list.append(np.full(len(ts), g["district_code"], dtype=np.int64))
        s_list.append(np.full(len(ts), g["solar_code"], dtype=np.int64))
        w_list.append(np.full(len(ts), g["wind_code"], dtype=np.int64))
    return (np.concatenate(t_list), np.concatenate(d_list),
            np.concatenate(s_list), np.concatenate(w_list))


def build_valid_windows_test(lookback):
    d = np.load(os.path.join(INPUT_DIR, "test_district_codes.npy"))
    s = np.load(os.path.join(INPUT_DIR, "test_solar_codes.npy"))
    w = np.load(os.path.join(INPUT_DIR, "test_wind_codes.npy"))
    is_context = np.load(os.path.join(INPUT_DIR, "test_is_context.npy"))

    change = np.zeros(len(d), dtype=bool)
    change[0] = True
    change[1:] = (d[1:] != d[:-1]) | (s[1:] != s[:-1]) | (w[1:] != w[:-1])
    group_starts = np.where(change)[0]
    group_ends = np.append(group_starts[1:], len(d))

    t_list, d_list, s_list, w_list = [], [], [], []
    for start, end in zip(group_starts, group_ends):
        candidate_ts = np.arange(start, end, dtype=np.int64)
        candidate_ts = candidate_ts[candidate_ts - lookback + 1 >= start]
        candidate_ts = candidate_ts[~is_context[candidate_ts]]
        if len(candidate_ts) == 0:
            continue
        t_list.append(candidate_ts)
        d_list.append(np.full(len(candidate_ts), d[start], dtype=np.int64))
        s_list.append(np.full(len(candidate_ts), s[start], dtype=np.int64))
        w_list.append(np.full(len(candidate_ts), w[start], dtype=np.int64))
    return (np.concatenate(t_list), np.concatenate(d_list),
            np.concatenate(s_list), np.concatenate(w_list))


class SequenceDataset(Dataset):
    def __init__(self, features, targets, t_positions, district_codes, solar_codes, wind_codes, lookback):
        self.features = features
        self.targets = targets
        self.t_positions = t_positions
        self.district_codes = district_codes
        self.solar_codes = solar_codes
        self.wind_codes = wind_codes
        self.lookback = lookback

    def __len__(self):
        return len(self.t_positions)

    def __getitem__(self, idx):
        t = self.t_positions[idx]
        window = np.array(self.features[t - self.lookback + 1: t + 1], dtype=np.float32)
        window = (window - FEATURE_MEAN) / FEATURE_STD
        target = np.array(self.targets[t], dtype=np.float32)
        target = (target - TARGET_MEAN) / TARGET_STD
        return (
            torch.from_numpy(window),
            torch.tensor(self.district_codes[idx], dtype=torch.long),
            torch.tensor(self.solar_codes[idx], dtype=torch.long),
            torch.tensor(self.wind_codes[idx], dtype=torch.long),
            torch.from_numpy(target),
        )


class CoverageSampler(Sampler):
    """Shuffles the full pool of valid training windows once, then hands
    out consecutive slices each epoch so every row is used exactly once
    before any row repeats - see module docstring for the full
    explanation of why this replaces plain random-with-replacement
    sampling. state_dict()/load_state_dict() let its position survive a
    checkpoint save/resume, so coverage continues seamlessly."""

    def __init__(self, n_total, samples_per_epoch, seed=0):
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

    def state_dict(self):
        return {"permutation": self.permutation, "cursor": self.cursor}

    def load_state_dict(self, state):
        self.permutation = state["permutation"]
        self.cursor = state["cursor"]


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
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
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


def masked_weighted_mse_loss(pred, target, weight_vector):
    """
    Skips NaN target entries when computing the loss (rows near the end
    of a group's series are missing some horizon targets - normal and
    expected).

    IMPORTANT SUBTLETY THIS FUNCTION HANDLES: it's not enough to just
    exclude NaN entries from the final SUM. PyTorch's autograd computes a
    gradient for (pred - target)**2 at EVERY position, including ones
    that get masked out afterward - and NaN * 0 is still NaN, not 0. So a
    single masked-out NaN-target position can silently poison the
    gradient for the whole batch, even though it never shows up in the
    visible loss value. This was the actual cause of the "NaN loss from
    step 2 onward" problem - not a precision/AMP issue at all, since it
    happens identically in FP32.

    THE FIX: replace NaN target entries with the corresponding prediction
    (detached, so no gradient flows through the replacement itself)
    BEFORE computing the squared error. This makes every position finite,
    so autograd never touches an actual NaN. The masking step below still
    correctly excludes those (now finite, but fake) entries from the
    final loss value and gives them a clean, genuine zero gradient.
    """
    valid = ~torch.isnan(target)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    target_safe = torch.where(valid, target, pred.detach())

    w = weight_vector.unsqueeze(0).expand_as(target)
    squared_error = (pred - target_safe) ** 2
    weighted_error = squared_error * w

    return weighted_error[valid].sum() / w[valid].sum()


def main():
    print("Loading train memmap arrays...")
    features_train = np.memmap(os.path.join(INPUT_DIR, "features_train.dat"), dtype="float32",
                                mode="r", shape=(TRAIN_META["n_rows"], N_FEATURES))
    targets_train = np.memmap(os.path.join(INPUT_DIR, "targets_train.dat"), dtype="float32",
                               mode="r", shape=(TRAIN_META["n_rows"], N_TARGETS))

    print("Loading test arrays...")
    features_test = np.load(os.path.join(INPUT_DIR, "features_test.npy"))
    targets_test = np.load(os.path.join(INPUT_DIR, "targets_test.npy"))

    print("Building valid window index (train)...")
    train_t, train_d, train_s, train_w = build_valid_windows_train(TRAIN_META["groups"], LOOKBACK_HOURS)
    print(f"  {len(train_t):,} valid training windows")

    print("Building valid window index (test)...")
    test_t, test_d, test_s, test_w = build_valid_windows_test(LOOKBACK_HOURS)
    print(f"  {len(test_t):,} valid evaluation windows")

    train_dataset = SequenceDataset(features_train, targets_train, train_t, train_d, train_s, train_w, LOOKBACK_HOURS)
    test_dataset = SequenceDataset(features_test, targets_test, test_t, test_d, test_s, test_w, LOOKBACK_HOURS)

    train_sampler = CoverageSampler(len(train_dataset), BATCH_SIZE * STEPS_PER_EPOCH, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                               num_workers=NUM_WORKERS, pin_memory=True)

    rng = np.random.default_rng(42)
    quick_val_idx = rng.choice(len(test_dataset), size=min(20000, len(test_dataset)), replace=False)
    quick_val_loader = DataLoader(Subset(test_dataset, quick_val_idx),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = GRUForecastNet(N_FEATURES, N_DISTRICTS, N_SOLAR_SIZES, N_WIND_SIZES,
                            HIDDEN_SIZE, EMBED_DIM_DISTRICT, EMBED_DIM_SIZE, N_HORIZONS).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    start_epoch = 0
    best_val_loss = float("inf")

    global_step = 0   # counts successful weight updates only, used to drive the LR warmup

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n[INFO] Found existing checkpoint at {CHECKPOINT_PATH}! Resuming.")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        train_sampler.load_state_dict(ckpt["sampler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        global_step = ckpt.get("global_step", 0)
        print(f"Resuming from epoch {start_epoch + 1}, best val loss so far: {best_val_loss:.4f}\n")

    training_run_start = time.time()   # tracks elapsed time for THIS run only (resets each time you launch the script)

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for step, (window, d_code, s_code, w_code, target) in enumerate(train_loader):
            window, d_code, s_code, w_code, target = (
                window.to(DEVICE), d_code.to(DEVICE), s_code.to(DEVICE),
                w_code.to(DEVICE), target.to(DEVICE),
            )

            optimizer.zero_grad()

            pred = model(window, d_code, s_code, w_code)
            loss = masked_weighted_mse_loss(pred, target, TARGET_WEIGHTS_TENSOR)

            if not torch.isfinite(loss):
                # A NaN/Inf loss would otherwise corrupt every model weight the
                # instant optimizer.step() runs, permanently - and it never
                # recovers on its own afterward. Skipping this one step costs
                # almost nothing, but protects the entire rest of the run.
                print(f"  [WARNING] Non-finite loss at epoch {epoch + 1} step {step + 1} "
                      f"- skipping this step, model weights left untouched.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Linear LR warmup for the first WARMUP_STEPS successful updates:
            # ramps the learning rate up gently from near-zero instead of
            # applying the full learning rate to a freshly-initialized model.
            # Random initial weights + a full-strength update is exactly the
            # combination most likely to push a model into an unstable
            # region early in training - this smooths that out.
            if global_step < WARMUP_STEPS:
                warmup_lr = LEARNING_RATE * (global_step + 1) / WARMUP_STEPS
                for group in optimizer.param_groups:
                    group["lr"] = warmup_lr

            optimizer.step()
            global_step += 1

            running_loss += loss.item()

            if (step + 1) % 400 == 0:
                print(f"  Epoch {epoch + 1} step {step + 1}/{STEPS_PER_EPOCH} "
                      f"- running loss: {running_loss / (step + 1):.4f}")

        avg_train_loss = running_loss / STEPS_PER_EPOCH

        model.eval()
        val_loss_total, val_batches = 0.0, 0
        with torch.no_grad():
            for window, d_code, s_code, w_code, target in quick_val_loader:
                window, d_code, s_code, w_code, target = (
                    window.to(DEVICE), d_code.to(DEVICE), s_code.to(DEVICE),
                    w_code.to(DEVICE), target.to(DEVICE),
                )
                pred = model(window, d_code, s_code, w_code)
                loss = masked_weighted_mse_loss(pred, target, TARGET_WEIGHTS_TENSOR)
                val_loss_total += loss.item()
                val_batches += 1
        avg_val_loss = val_loss_total / val_batches

        scheduler.step(avg_val_loss)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} done in {elapsed / 60:.1f} min - "
              f"train loss: {avg_train_loss:.4f} - quick val loss: {avg_val_loss:.4f}")

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "sampler_state": train_sampler.state_dict(),
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        }, CHECKPOINT_PATH)
        print(f"  Checkpoint saved -> {CHECKPOINT_PATH}")

        if is_best:
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            print(f"  *** New best model saved! *** -> {FINAL_MODEL_PATH}")

        run_elapsed_hours = (time.time() - training_run_start) / 3600
        print(f"  Time elapsed this run: {run_elapsed_hours:.2f}h / {TRAIN_TIME_BUDGET_HOURS}h budget")
        if run_elapsed_hours >= TRAIN_TIME_BUDGET_HOURS:
            print(f"\nTime budget of {TRAIN_TIME_BUDGET_HOURS} hours reached after epoch {epoch + 1}. "
                  f"Stopping cleanly - the last checkpoint above is fully valid and resumable.")
            break

    print("\n===========================================")
    print("TRAINING COMPLETE (v2: climatology + 336h lookback + coverage sampling + horizon weighting)")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Final model saved at: {FINAL_MODEL_PATH}")
    print("===========================================")


if __name__ == "__main__":
    main()