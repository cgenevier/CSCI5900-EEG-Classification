from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import mne

# -------------------------
# Config
# -------------------------
RAW_FIF_ROOT = Path("datasets/bnci_horizon_2020_ErrP/raw_fif")
TASK = "errp"

# If you only want to inspect one subject/session/run, set these to strings like "01"
ONLY_SUBJECT = None   # e.g., "01"
ONLY_SESSION = None   # e.g., "01"
ONLY_RUN = None       # e.g., "01"

# Plot toggles (leave False for headless / CI)
SHOW_RAW_PLOT = False
SHOW_PSD_PLOT = False
SHOW_EVENTS_PLOT = False

# How many seconds of raw to display when plotting
RAW_PLOT_DURATION_SEC = 10.0

# Event codes you expect in this dataset (BNCI ErrP)
EXPECTED_EVENT_CODES = {1, 2, 4, 5, 6, 8, 9, 10}

# Amplitude QC window (seconds from start of recording)
AMP_QC_WINDOW_SEC = 5.0

# Amplitude QC is computed after per-channel demeaning (removes full-DC offsets).
# Thresholds are intentionally generous; they're meant to catch unit mistakes / extreme issues.
DEMEANED_MEDIAN_WARN_V = 1e-3     # 1000 µV median is suspicious
DEMEANED_P999_WARN_V = 2e-3       # 2000 µV p99.9 is suspicious
DEMEANED_MAX_WARN_V = 1e-2        # 10 mV max spike (after demeaning) is noteworthy


# -------------------------
# Helpers
# -------------------------
def iter_fif_files(root: Path):
    pattern = f"**/sub-*_ses-*_task-{TASK}_run-*_raw.fif"
    for p in sorted(root.glob(pattern)):
        if ONLY_SUBJECT and f"sub-{ONLY_SUBJECT}" not in str(p):
            continue
        if ONLY_SESSION and f"ses-{ONLY_SESSION}" not in str(p):
            continue
        if ONLY_RUN and f"run-{ONLY_RUN}" not in p.name:
            continue
        yield p


def find_sidecars(fif_path: Path):
    base = fif_path.name.replace("_raw.fif", "")
    events_csv = fif_path.parent / f"{base}_events.csv"
    meta_json = fif_path.parent / f"{base}_meta.json"
    return events_csv, meta_json


def load_events_csv(events_csv: Path) -> pd.DataFrame:
    if not events_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(events_csv)


def load_meta_json(meta_json: Path) -> dict:
    if not meta_json.exists():
        return {}
    with open(meta_json, "r") as f:
        return json.load(f)


def events_df_to_mne_events(events_df: pd.DataFrame, sample_col: str = "sample0") -> np.ndarray:
    if events_df is None or len(events_df) == 0:
        return np.zeros((0, 3), dtype=int)
    return np.column_stack(
        [
            events_df[sample_col].astype(int).to_numpy(),
            np.zeros(len(events_df), dtype=int),
            events_df["code"].astype(int).to_numpy(),
        ]
    ).astype(int)


def demean_per_channel(x: np.ndarray) -> np.ndarray:
    """x shape: (n_channels, n_times)"""
    if x.size == 0:
        return x
    return x - x.mean(axis=1, keepdims=True)


def amplitude_stats_demeaned(raw: mne.io.BaseRaw, window_sec: float) -> dict:
    """
    Compute robust amplitude stats on EEG channels, after demeaning per channel.
    Uses first `window_sec` seconds to avoid loading full file.
    """
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)

    sample_len = int(min(n_times, max(1, int(sfreq * window_sec))))
    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, misc=False)

    if len(picks) == 0:
        return {
            "demeaned_median_abs_v_firstwin": float("nan"),
            "demeaned_p999_abs_v_firstwin": float("nan"),
            "demeaned_max_abs_v_firstwin": float("nan"),
            "demeaned_max_channel_firstwin": None,
        }

    data, _ = raw[picks, :sample_len]  # (n_eeg, sample_len)
    data = demean_per_channel(data)
    abs_data = np.abs(data)

    med = float(np.median(abs_data))
    p999 = float(np.percentile(abs_data, 99.9))
    max_abs = float(np.max(abs_data))

    # Identify which EEG channel produced the max (within this window)
    flat = int(np.argmax(abs_data))
    ch_i, _t_i = np.unravel_index(flat, abs_data.shape)
    max_ch_name = raw.ch_names[picks[ch_i]]

    return {
        "demeaned_median_abs_v_firstwin": med,
        "demeaned_p999_abs_v_firstwin": p999,
        "demeaned_max_abs_v_firstwin": max_abs,
        "demeaned_max_channel_firstwin": max_ch_name,
    }


# -------------------------
# QC checks
# -------------------------
def qc_one_file(fif_path: Path) -> dict:
    events_csv, meta_json = find_sidecars(fif_path)
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")

    # Basic info
    nchan = int(raw.info["nchan"])
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    duration_sec = n_times / sfreq if sfreq else float("nan")
    ch_names = raw.ch_names
    ch_types = set(raw.get_channel_types())

    # Sidecars
    meta = load_meta_json(meta_json)
    events_df = load_events_csv(events_csv)

    # Checks: channel names unique
    unique_ok = (len(set(ch_names)) == len(ch_names))

    # Checks: expected channel types
    eeg_count = len(mne.pick_types(raw.info, eeg=True, meg=False))
    has_only_eeg = (ch_types == {"eeg"})  # expected for your converted BNCI exports

    # Amplitude sanity on demeaned EEG (robust to full-DC offsets)
    amp = amplitude_stats_demeaned(raw, window_sec=AMP_QC_WINDOW_SEC)

    # Events checks
    codes_present = set()
    in_bounds_ok = True
    if len(events_df) > 0:
        codes_present = set(map(int, events_df["code"].unique()))
        in_bounds_ok = bool(((events_df["sample0"] >= 0) & (events_df["sample0"] < n_times)).all())
    expected_codes_ok = (codes_present.issubset(EXPECTED_EVENT_CODES)) if codes_present else True

    report = {
        "fif": str(fif_path),
        "sfreq": sfreq,
        "nchan": nchan,
        "eeg_count": eeg_count,
        "n_times": n_times,
        "duration_sec": duration_sec,
        "unique_ch_names": unique_ok,
        "channel_types": sorted(list(ch_types)),
        "only_eeg_types": has_only_eeg,
        # Robust amplitude QC (demeaned)
        **amp,
        "events_csv_exists": events_csv.exists(),
        "meta_json_exists": meta_json.exists(),
        "n_events": int(len(events_df)) if len(events_df) else 0,
        "codes_present": sorted(list(codes_present)) if codes_present else [],
        "expected_codes_subset": expected_codes_ok,
        "events_in_bounds": in_bounds_ok,
        "meta_input_units": meta.get("input_units"),
        "meta_stored_units": meta.get("stored_units"),
    }

    # Optional plots (interactive)
    if SHOW_RAW_PLOT:
        raw.plot(duration=RAW_PLOT_DURATION_SEC, n_channels=min(20, nchan), scalings="auto", show=True)

    if SHOW_PSD_PLOT:
        raw.compute_psd(fmin=0.1, fmax=40).plot(show=True)

    if SHOW_EVENTS_PLOT and len(events_df) > 0:
        events = events_df_to_mne_events(events_df, sample_col="sample0")
        mne.viz.plot_events(events, sfreq=sfreq, first_samp=0, show=True)

    return report


def main():
    fif_files = list(iter_fif_files(RAW_FIF_ROOT))
    if not fif_files:
        raise FileNotFoundError(f"No FIF files found under {RAW_FIF_ROOT}")

    print(f"Found {len(fif_files)} FIF files under: {RAW_FIF_ROOT}\n")

    reports = []
    for p in fif_files:
        rep = qc_one_file(p)
        reports.append(rep)

        # Quick per-file summary line
        print(
            f"{Path(rep['fif']).name} | "
            f"{rep['nchan']} ch | {rep['sfreq']} Hz | {rep['duration_sec']:.1f}s | "
            f"{rep['n_events']} events | codes={rep['codes_present']}"
        )

        # Flag obvious problems
        problems = []
        if not rep["unique_ch_names"]:
            problems.append("duplicate channel names")
        if not rep["only_eeg_types"]:
            problems.append(f"unexpected channel types {rep['channel_types']}")
        if not rep["events_csv_exists"]:
            problems.append("missing events_csv")
        if not rep["meta_json_exists"]:
            problems.append("missing meta_json")
        if not rep["events_in_bounds"]:
            problems.append("events out of bounds")
        if not rep["expected_codes_subset"]:
            problems.append(f"unexpected event codes {rep['codes_present']}")

        # Amplitude checks (demeaned) — robust to full DC offsets
        med = rep["demeaned_median_abs_v_firstwin"]
        p999 = rep["demeaned_p999_abs_v_firstwin"]
        mx = rep["demeaned_max_abs_v_firstwin"]
        mx_ch = rep["demeaned_max_channel_firstwin"]

        if np.isfinite(med) and med > DEMEANED_MEDIAN_WARN_V:
            problems.append(f"demeaned median high ({med:.2e} V)")
        if np.isfinite(p999) and p999 > DEMEANED_P999_WARN_V:
            problems.append(f"demeaned p99.9 high ({p999:.2e} V)")
        if np.isfinite(mx) and mx > DEMEANED_MAX_WARN_V:
            problems.append(f"demeaned max spike ({mx:.2e} V @ {mx_ch})")

        if problems:
            print("  ⚠️  QC flags:", "; ".join(problems))

    # Write a QC summary CSV for easy scanning
    out_csv = Path("datasets/bnci_horizon_2020_ErrP/qc_summary.csv")
    pd.DataFrame(reports).to_csv(out_csv, index=False)
    print(f"\nWrote QC summary: {out_csv.resolve()}")


if __name__ == "__main__":
    main()