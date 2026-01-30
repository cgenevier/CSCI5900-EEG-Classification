"""
Preprocess BNCI Horizon 2020 ErrP raw FIF runs into multiple variants.

Outputs (per run):
  1) (after dropping flagged channels) CAR + causal FIR band-pass 0.1–30 Hz + resample 256 Hz
  2) (after dropping flagged channels) CAR + causal FIR band-pass 0.3–30 Hz + resample 256 Hz
  3) (after dropping flagged channels) CAR + causal FIR band-pass 1–10 Hz   + resample 256 Hz
  4) Same as (2) + ICA-cleaned (ICA fit on 1–30 Hz copy; applied to 0.3–30 Hz data)

Saves to: datasets/bnci_horizon_2020_ErrP/preprocessed_fif/<variant>/sub-xx/ses-yy/<base>_raw.fif
Also saves ICA solutions (for variant 4) to: .../<variant>/sub-xx/ses-yy/<base>-ica.fif

QA integration:
  - Reads datasets/bnci_horizon_2020_ErrP/qc_outputs/qc_channel_stats.csv
  - Drops *flagged* Fp1 channels (or optionally all flagged channels) BEFORE CAR.
"""

from __future__ import annotations

from pathlib import Path
import mne
import numpy as np
import pandas as pd

# -------------------------
# Config
# -------------------------
RAW_FIF_ROOT = Path("datasets/bnci_horizon_2020_ErrP/raw_fif")
TASK = "errp"

OUT_ROOT = Path("datasets/bnci_horizon_2020_ErrP/preprocessed_fif")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# QC outputs (from your QA script)
QC_OUT_ROOT = Path("datasets/bnci_horizon_2020_ErrP/qc_outputs")
QC_CHANNEL_CSV = QC_OUT_ROOT / "qc_channel_stats.csv"

# Optional subset controls
ONLY_SUBJECT = None   # e.g., "01"
ONLY_SESSION = None   # e.g., "01"
ONLY_RUN = None       # e.g., "01"

# Target sampling rate
TARGET_SFREQ = 256.0

# Filtering options (causal FIR)
BANDPASSES = {
    "car_fir_causal_bp-0.1-30_sfreq256": (0.1, 30.0),
    "car_fir_causal_bp-0.3-30_sfreq256": (0.3, 30.0),
    "car_fir_causal_bp-1-10_sfreq256": (1.0, 10.0),
}

# ICA option (applied to the 0.3–30 dataset)
ICA_VARIANT = "car_fir_causal_bp-0.3-30_sfreq256_ica"

# ICA fitting parameters
ICA_METHOD = "infomax"          # good default for EEG
ICA_N_COMPONENTS = 0.99         # keep 99% variance
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = "auto"

# ICA EOG proxy channel preference (dataset has EEG only)
EOG_PROXY_CANDIDATES = ["Fp1", "Fp2", "AF7", "AF8", "AF3", "AF4"]

# -------------------------
# QA-based channel dropping policy (BEFORE CAR)
# -------------------------
# If True: drop ALL channels flagged by QA (any_flag==True) for this run.
# If False: only drop channels in DROP_ONLY_THESE_IF_FLAGGED when they were flagged.
DROP_ALL_FLAGGED_CHANNELS = True

# Your request: example: drop flagged Fp1 channels then set {'Fp1'}
DROP_ONLY_THESE_IF_FLAGGED = {}

# -------------------------
# Helpers
# -------------------------
def iter_fif_files(root: Path):
    pattern = f"**/sub-*_ses-*_task-{TASK}_run-*_raw.fif"
    for p in sorted(root.glob(pattern)):
        sp = str(p)
        if ONLY_SUBJECT and f"sub-{ONLY_SUBJECT}" not in sp:
            continue
        if ONLY_SESSION and f"ses-{ONLY_SESSION}" not in sp:
            continue
        if ONLY_RUN and f"run-{ONLY_RUN}" not in p.name:
            continue
        yield p


def parse_bids_bits(fif_path: Path) -> tuple[str, str, str]:
    """
    Extract (sub-XX, ses-YY, base) from filename/path.
    base = file stem without "_raw.fif"
    """
    base = fif_path.name.replace("_raw.fif", "")
    parts = fif_path.parts
    sub = next((x for x in parts if x.startswith("sub-")), "sub-unk")
    ses = next((x for x in parts if x.startswith("ses-")), "ses-unk")
    return sub, ses, base


def ensure_eeg_only(raw: mne.io.BaseRaw) -> None:
    """Warn if unexpected channel types exist."""
    types = set(raw.get_channel_types())
    if types != {"eeg"}:
        print(f"⚠️ Channel types present: {sorted(types)} (expected only EEG). Proceeding anyway.")


def apply_car(raw: mne.io.BaseRaw) -> None:
    """Common average reference across EEG channels."""
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")


def apply_causal_fir_bandpass(raw: mne.io.BaseRaw, l_freq: float, h_freq: float) -> None:
    """Causal FIR band-pass (options for phase: linear, minimum, minimum-half)."""
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        phase="linear",
        fir_design="firwin",
        verbose="ERROR",
    )


def resample_to(raw: mne.io.BaseRaw, sfreq: float) -> None:
    """Resample with MNE's built-in anti-alias filtering."""
    raw.resample(sfreq, npad="auto", verbose="ERROR")


def save_raw_variant(raw: mne.io.BaseRaw, variant_dir: Path, sub: str, ses: str, base: str) -> Path:
    out_dir = variant_dir / sub / ses
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}_raw.fif"
    raw.save(out_path, overwrite=True, verbose="ERROR")
    return out_path


def pick_eog_proxy_channel(raw: mne.io.BaseRaw) -> str | None:
    chset = set(raw.ch_names)
    for name in EOG_PROXY_CANDIDATES:
        if name in chset:
            return name
    return None


def load_flagged_channels_map(qc_channel_csv: Path) -> dict[str, list[str]]:
    """
    Returns a dict mapping fif_base -> list of channels flagged by QA.
    Requires columns: fif_base, ch_name, any_flag
    """
    if not qc_channel_csv.exists():
        raise FileNotFoundError(f"QC channel CSV not found: {qc_channel_csv}")

    df = pd.read_csv(qc_channel_csv)

    required = {"fif_base", "ch_name", "any_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"QC CSV missing columns: {missing}. Found: {df.columns.tolist()}")

    df_flag = df[df["any_flag"] == True].copy()

    out: dict[str, list[str]] = {}
    for base, g in df_flag.groupby("fif_base"):
        out[base] = g["ch_name"].astype(str).tolist()
    return out


def drop_flagged_channels_pre_car(raw: mne.io.BaseRaw, fif_base: str, flagged_map: dict[str, list[str]]) -> list[str]:
    """
    Drops selected channels (based on QA flags) BEFORE referencing.
    Returns list of channels actually dropped.
    """
    flagged = flagged_map.get(fif_base, [])
    if not flagged:
        return []

    if DROP_ALL_FLAGGED_CHANNELS:
        to_drop = set(flagged)
    else:
        to_drop = set(ch for ch in flagged if ch in DROP_ONLY_THESE_IF_FLAGGED)

    # Only drop channels that are present
    to_drop_list = [ch for ch in sorted(to_drop) if ch in raw.ch_names]
    if not to_drop_list:
        return []

    raw.drop_channels(to_drop_list)
    return to_drop_list


def fit_and_apply_ica_on_03_30(raw_03_30: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA, list[int]]:
    """
    Fit ICA on a 1–30 Hz high-passed copy (for stability), then apply it to a copy
    of raw_03_30 (0.3–30 Hz). Uses a frontal EEG channel as an EOG proxy if no
    true EOG channels exist.

    Returns:
      raw_ica:  ICA-cleaned copy of raw_03_30
      ica:      fitted ICA object (with exclude set)
      exclude:  list of excluded component indices
    """
    from mne.preprocessing import ICA

    # 1) Fit copy: 1–30 Hz (causal FIR) improves ICA stability vs 0.3 Hz HPF
    raw_fit = raw_03_30.copy()
    raw_fit.filter(
        l_freq=1.0,
        h_freq=30.0,
        method="fir",
        phase="linear",
        fir_design="firwin",
        verbose="ERROR",
    )

    # 2) Fit ICA
    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
    )
    ica.fit(raw_fit, verbose="ERROR")

    # 3) Auto-detect blink-ish components using a proxy channel (EEG-only data)
    proxy = pick_eog_proxy_channel(raw_fit)
    exclude: list[int] = []
    if proxy is not None:
        eog_inds, eog_scores = ica.find_bads_eog(raw_fit, ch_name=proxy, verbose="ERROR")
        exclude = list(eog_inds)
        ica.exclude = exclude
    else:
        print("⚠️ No frontal proxy channel (Fp1/Fp2/AF*) found; ICA fit done, nothing auto-excluded.")

    # 4) Apply ICA to your target 0.3–30 data
    raw_ica = raw_03_30.copy()
    ica.apply(raw_ica, verbose="ERROR")

    return raw_ica, ica, exclude


def preprocess_one_file(fif_path: Path, flagged_map: dict[str, list[str]]) -> None:
    sub, ses, base = parse_bids_bits(fif_path)
    print(f"\n=== {base} ===")

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    ensure_eeg_only(raw)

    # 0) Drop QA-flagged channels BEFORE CAR/filtering
    dropped = drop_flagged_channels_pre_car(raw, base, flagged_map)
    if dropped:
        print(f"  Dropped pre-CAR flagged channels: {dropped}")

    # Create each variant from the same starting point (post-drop).
    for variant_name, (l_freq, h_freq) in BANDPASSES.items():
        variant_dir = OUT_ROOT / variant_name

        r = raw.copy()
        # 1) CAR
        apply_car(r)

        # 2) Causal FIR band-pass
        apply_causal_fir_bandpass(r, l_freq=l_freq, h_freq=h_freq)

        # 3) Downsample
        resample_to(r, TARGET_SFREQ)

        out_path = save_raw_variant(r, variant_dir, sub, ses, base)
        print(f"  Saved: {out_path}")

    # ICA variant built from the 0.3–30 pipeline (post-drop)
    r_03_30 = raw.copy()
    apply_car(r_03_30)
    apply_causal_fir_bandpass(r_03_30, l_freq=0.3, h_freq=30.0)
    resample_to(r_03_30, TARGET_SFREQ)

    raw_ica, ica, excluded = fit_and_apply_ica_on_03_30(r_03_30)

    ica_dir = OUT_ROOT / ICA_VARIANT
    out_path_ica = save_raw_variant(raw_ica, ica_dir, sub, ses, base)
    ica_path = (ica_dir / sub / ses / f"{base}-ica.fif")
    ica.save(ica_path, overwrite=True)

    print(f"  Saved ICA-cleaned: {out_path_ica}")
    print(f"  Saved ICA object:  {ica_path}")
    if excluded:
        print(f"  ICA excluded components (auto via proxy): {excluded}")
    else:
        print("  ICA excluded components: none (auto-detection found none or no proxy channel).")


def main():
    # Load QA-flagged channels once
    flagged_map = load_flagged_channels_map(QC_CHANNEL_CSV)

    fif_files = list(iter_fif_files(RAW_FIF_ROOT))
    if not fif_files:
        raise FileNotFoundError(f"No raw FIF files found under: {RAW_FIF_ROOT}")

    print(f"Found {len(fif_files)} raw FIF files under: {RAW_FIF_ROOT}")
    print(f"Will write preprocessed FIFs to: {OUT_ROOT.resolve()}")
    print(f"Using QA flags from: {QC_CHANNEL_CSV.resolve()}")
    if DROP_ALL_FLAGGED_CHANNELS:
        print("Channel drop policy: DROP ALL QA-flagged channels before CAR.")
    else:
        print(f"Channel drop policy: DROP only these if QA-flagged: {sorted(DROP_ONLY_THESE_IF_FLAGGED)}")

    for p in fif_files:
        preprocess_one_file(p, flagged_map)


if __name__ == "__main__":
    main()