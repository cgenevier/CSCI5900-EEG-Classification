"""
Epoch QC / rejection script (artifact correction/rejection)

Reads:  datasets/bnci_horizon_2020_ErrP/epoched_fif/**/**/**/**/*-epo.fif*
Writes: datasets/bnci_horizon_2020_ErrP/epoched_fif_clean/<same structure>/*-epo-clean.fif

Also writes a per-run QC CSV alongside the cleaned epochs:
  .../*-epo-qc.csv

Rejection methods included:
  1) MNE-native drop_bad using peak-to-peak (reject) + flat thresholds
  2) Optional robust outlier rejection on per-epoch RMS (variance-ish) metric

Notes:
  - Thresholds are in Volts (e.g., 150e-6 = 150 µV)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import mne


# -------------------------
# Config
# -------------------------
EPOCHS_ROOT = Path("datasets/bnci_horizon_2020_ErrP/epoched_fif")
OUT_ROOT    = Path("datasets/bnci_horizon_2020_ErrP/epoched_fif_clean")

# MNE drop_bad thresholds (peak-to-peak + flat), in Volts
REJECT = dict(eeg=200e-6)  # start point: 200 µV p2p
FLAT   = dict(eeg=1e-6)    # start point: 1 µV too-flat

# Optional robust outlier rejection on epoch RMS (after drop_bad).
ENABLE_RMS_OUTLIERS = False # Setting this to False for now to be conservative
RMS_Z_CUTOFF = 6.0  # robust z-score cutoff (MAD-based). 5–8 is typical.
RMS_TOP_PCT = None  # alternative: e.g. 0.01 to drop top 1% by RMS; set None to disable.
RMS_PICKS = "eeg" # Only consider these channel types for RMS metric


# -------------------------
# Helpers
# -------------------------
def iter_epochs_files(root: Path):
    # matches .fif and .fif.gz
    return sorted(root.rglob("*-epo.fif*"))


def parse_structure(epo_path: Path) -> tuple[Path, str]:
    """
    Returns:
      rel_dir: path under epoched_fif (e.g., epochVariant/preprocVariant/sub-xx/ses-yy)
      base: filename base without -epo.fif(.gz)
    """
    parts = epo_path.parts
    try:
        idx = parts.index("epoched_fif")
    except ValueError:
        # fallback: mirror full parent structure if unexpected
        rel_dir = epo_path.parent
    else:
        rel_dir = Path(*parts[idx + 1 : -1])  # everything after epoched_fif up to parent dir

    name = epo_path.name
    if name.endswith("-epo.fif.gz"):
        base = name[:-len("-epo.fif.gz")]
    elif name.endswith("-epo.fif"):
        base = name[:-len("-epo.fif")]
    else:
        base = epo_path.stem.replace("-epo", "")

    return rel_dir, base


def robust_z(x: np.ndarray) -> np.ndarray:
    """
    Robust z-score using MAD.
    Returns z where z ~ 0 for median; scale is 1.4826 * MAD.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale == 0:
        # all equal -> no outliers
        return np.zeros_like(x)
    return (x - med) / scale


def epoch_rms_metric(epochs: mne.Epochs, picks=RMS_PICKS) -> np.ndarray:
    """
    Per-epoch RMS computed across channels and time:
      rms_i = sqrt(mean(x_i^2))
    Shape returned: (n_epochs,)
    """
    data = epochs.copy().pick(picks).get_data()  # (n_epochs, n_ch, n_times)
    return np.sqrt(np.mean(data ** 2, axis=(1, 2)))


# -------------------------
# Main
# -------------------------
def main():
    epo_files = iter_epochs_files(EPOCHS_ROOT)
    if not epo_files:
        raise FileNotFoundError(f"No epoched FIF files found under: {EPOCHS_ROOT.resolve()}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(epo_files)} epoch files under: {EPOCHS_ROOT.resolve()}")
    print(f"Writing cleaned epochs under: {OUT_ROOT.resolve()}")

    for epo_path in epo_files:
        rel_dir, base = parse_structure(epo_path)
        print(f"\nQC: {base}")

        epochs = mne.read_epochs(epo_path, preload=True, verbose="ERROR")
        n0 = len(epochs)

        # 1) MNE-native rejection
        epochs_clean = epochs.copy()
        epochs_clean.drop_bad(reject=REJECT, flat=FLAT, verbose="ERROR")
        n1 = len(epochs_clean)

        # 2) Optional RMS outlier rejection
        rms_z = None
        rms = None
        dropped_rms = []
        if ENABLE_RMS_OUTLIERS and len(epochs_clean) > 0:
            rms = epoch_rms_metric(epochs_clean, picks=RMS_PICKS)
            if RMS_TOP_PCT is not None:
                # Drop top fraction by RMS
                k = int(np.ceil(len(rms) * float(RMS_TOP_PCT)))
                if k > 0:
                    idx_sorted = np.argsort(rms)  # ascending
                    drop_local = idx_sorted[-k:]
                    epochs_clean.drop(drop_local.tolist(), reason="RMS_OUTLIER")
                    dropped_rms = drop_local.tolist()
            else:
                rms_z = robust_z(rms)
                drop_local = np.where(rms_z > float(RMS_Z_CUTOFF))[0]
                if drop_local.size > 0:
                    epochs_clean.drop(drop_local.tolist(), reason="RMS_OUTLIER")
                    dropped_rms = drop_local.tolist()

        n2 = len(epochs_clean)

        # Write outputs
        out_dir = OUT_ROOT / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        out_epo = out_dir / f"{base}-epo-clean.fif"
        epochs_clean.save(out_epo, overwrite=True)
        print(f"  Saved cleaned epochs: {out_epo}  ({n0} → {n2})")

        # QC report CSV (one row, per run)
        qc = {
            "input_path": str(epo_path),
            "output_path": str(out_epo),
            "n_epochs_in": int(n0),
            "n_after_drop_bad": int(n1),
            "n_epochs_out": int(n2),
            "n_dropped_drop_bad": int(n0 - n1),
            "n_dropped_rms_outlier": int(n1 - n2),
            "reject_eeg_v": float(REJECT.get("eeg", np.nan)),
            "flat_eeg_v": float(FLAT.get("eeg", np.nan)),
            "rms_outliers_enabled": bool(ENABLE_RMS_OUTLIERS),
            "rms_z_cutoff": float(RMS_Z_CUTOFF) if RMS_TOP_PCT is None else np.nan,
            "rms_top_pct": float(RMS_TOP_PCT) if RMS_TOP_PCT is not None else np.nan,
        }

        # add some summary stats if available
        if rms is not None and len(rms) > 0:
            qc.update(
                rms_median=float(np.median(rms)),
                rms_p95=float(np.percentile(rms, 95)),
                rms_max=float(np.max(rms)),
                rms_outlier_indices=json.dumps(dropped_rms),
            )
        else:
            qc.update(
                rms_median=np.nan,
                rms_p95=np.nan,
                rms_max=np.nan,
                rms_outlier_indices="[]",
            )

        qc_path = out_dir / f"{base}-epo-qc.csv"
        pd.DataFrame([qc]).to_csv(qc_path, index=False)
        print(f"  Saved QC CSV: {qc_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()