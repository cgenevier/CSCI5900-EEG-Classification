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
  - Drops all flagged channels BEFORE CAR.
"""

from __future__ import annotations

from pathlib import Path
import mne
import mne.preprocessing as preprocessing
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

# Filtering Variations
VARIATIONS = {
    "car_fir_causal_bp-0.1-30": {
        "rereference": "average",
        "l_freq": 0.1,
        "h_freq": 30.0,
        "phase": "linear",
        "downsample": False,
        "target_sfreq": 256.0,
        "ica": False,
    },
    "car_fir_causal_bp-0.3-30": {
        "rereference": "average",
        "l_freq": 0.3,
        "h_freq": 30.0,
        "phase": "linear",
        "downsample": False,
        "target_sfreq": 256.0,
        "ica": False,
    },
    "car_fir_causal_bp-1-10": {
        "rereference": "average",
        "l_freq": 1.0,
        "h_freq": 10.0,
        "phase": "linear",
        "downsample": False,
        "target_sfreq": 256.0,
        "ica": False,
    },
    "car_fir_causal_bp-0.3-30_ica": {
        "rereference": "average",
        "l_freq": 0.3,
        "h_freq": 30.0,
        "phase": "linear",
        "downsample": False,
        "target_sfreq": 256.0,
        "ica": True,
        "ica_method": "fastica",
        "ica_n_components": 0.95,
        "ica_random_state": 97,
        "ica_max_iter": "auto",
        "ica_eog_proxy_candidates": ["Fp1", "Fp2", "AF7", "AF8", "AF3", "AF4"],
    },
}


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


def preprocess_one_file(fif_path: Path, flagged_map: dict[str, list[str]]) -> None:
    sub, ses, base = parse_bids_bits(fif_path)
    print(f"\n=== {base} ===")

    # Load raw data into MNE
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

    # Warn if unexpected channel types exist.
    types = set(raw.get_channel_types())
    if types != {"eeg"}:
        print(f"⚠️ Channel types present: {sorted(types)} (expected only EEG). Proceeding anyway.")

    # Drop QA-flagged channels BEFORE CAR/filtering
    flagged = flagged_map.get(base, [])
    raw.drop_channels(flagged)
    print(f"  Dropped pre-CAR flagged channels: {flagged}")

    # Create each variant from the same starting point (post-drop).
    for variant_name, variant_config in VARIATIONS.items():
        variant_dir = OUT_ROOT / variant_name
        out_dir = variant_dir / sub / ses
        out_dir.mkdir(parents=True, exist_ok=True)

        # create a copy of the raw data
        r = raw.copy()

        # 1) CAR
        r.set_eeg_reference(variant_config["rereference"], projection=False, verbose="ERROR")

        # 2) Causal FIR band-pass (phase options for causal FIR: linear, minimum, minimum-half).
        r.filter(l_freq=variant_config["l_freq"], h_freq=variant_config["h_freq"], method="fir", phase=variant_config["phase"], fir_design="firwin", verbose="ERROR")

        # 3) Downsample (MNE's built-in anti-alias filtering)
        if variant_config["downsample"]:
            r.resample(variant_config["target_sfreq"], npad="auto", verbose="ERROR")

        # 4) ICA
        if variant_config["ica"]:
            # a) Fit copy: 1–30 Hz (causal FIR) improves ICA stability vs 0.3 Hz HPF
            r_fit = r.copy()
            r_fit.filter(l_freq=1.0, h_freq=30.0, method="fir", phase="linear", fir_design="firwin", verbose="ERROR")

            # b) Fit ICA
            ica = preprocessing.ICA(n_components=variant_config["ica_n_components"], method=variant_config["ica_method"], random_state=variant_config["ica_random_state"], max_iter=variant_config["ica_max_iter"])
            ica.fit(r_fit, decim=3, verbose="ERROR")

            # c) Auto-detect blink-ish components using a proxy channel (EEG-only data)
            candidates = variant_config["ica_eog_proxy_candidates"]
            proxy = None
            for ch in candidates:
                if ch in r_fit.ch_names:
                    proxy = ch
                    break
            exclude: list[int] = []
            if proxy is not None:
                eog_inds, eog_scores = ica.find_bads_eog(r_fit, ch_name=proxy, verbose="ERROR")
                exclude = list(eog_inds)
            else:
                print("⚠️ No frontal proxy channel (Fp1/Fp2/AF*) found; ICA fit done, nothing auto-excluded.")

            # d) Apply ICA only if exclusions exist
            if exclude:
                ica.exclude = exclude
                ica.apply(r, verbose="ERROR")

                ica_path = out_dir / f"{base}-ica.fif"
                ica.save(ica_path)
                print(f"  ICA excluded components (auto via proxy): {exclude}")
            else:
                print("  ICA excluded components: none (auto-detection found none or no proxy channel).")

        # Save the preprocessed raw data
        out_path = out_dir / f"{base}_raw.fif"
        r.save(out_path, overwrite=True, verbose="ERROR")
        print(f"  Saved: {out_path}")


def main():
    # Load QA-flagged channels
    flagged_map: dict[str, list[str]] = {}
    if QC_CHANNEL_CSV.exists():
        df = pd.read_csv(QC_CHANNEL_CSV)
        required = {"fif_base", "ch_name", "any_flag"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"QC CSV missing columns: {missing}. Found: {df.columns.tolist()}")

        df_flag = df[df["any_flag"] == True].copy()
        flagged_map = {base: g["ch_name"].astype(str).tolist() for base, g in df_flag.groupby("fif_base")}

    fif_files = list(iter_fif_files(RAW_FIF_ROOT))
    if not fif_files:
        raise FileNotFoundError(f"No raw FIF files found under: {RAW_FIF_ROOT}")

    print(f"Found {len(fif_files)} raw FIF files under: {RAW_FIF_ROOT}")
    print(f"Will write preprocessed FIFs to: {OUT_ROOT.resolve()}")
    print(f"Using QA flags from: {QC_CHANNEL_CSV.resolve()}")

    for p in fif_files:
        preprocess_one_file(p, flagged_map)


if __name__ == "__main__":
    main()