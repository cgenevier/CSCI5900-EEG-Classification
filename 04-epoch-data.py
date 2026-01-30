from pathlib import Path
import mne

# -------------------------
# Config
# -------------------------
PREPROC_ROOT = Path("datasets/bnci_horizon_2020_ErrP/preprocessed_fif")
EPOCHS_ROOT  = Path("datasets/bnci_horizon_2020_ErrP/epoched_fif")

# Define epoch variants here (ms → seconds)
EPOCH_VARIANTS = {
    "tmin-200ms_tmax600ms": {"tmin": -0.200, "tmax": 0.600, "baseline": (-0.2, 0)},
    "tmin0ms_tmax800ms":   {"tmin":  0.000, "tmax": 0.800, "baseline": None},
}

# -------------------------
# Helpers
# -------------------------
def parse_bids_bits(fif_path: Path):
    parts = fif_path.parts
    variant = parts[parts.index("preprocessed_fif") + 1]
    sub = next(p for p in parts if p.startswith("sub-"))
    ses = next(p for p in parts if p.startswith("ses-"))
    base = fif_path.name.replace("_raw.fif", "").replace("_raw.fif.gz", "")
    return variant, sub, ses, base


# -------------------------
# Main
# -------------------------
def main():
    raws = sorted(PREPROC_ROOT.rglob("*_raw.fif*"))
    if not raws:
        raise FileNotFoundError(f"No preprocessed FIF files found under {PREPROC_ROOT}")

    print(f"Found {len(raws)} preprocessed raw FIF files")

    for raw_path in raws:
        variant, sub, ses, base = parse_bids_bits(raw_path)
        print(f"\nLoading: {base}")

        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")

        print("n_annotations:", len(raw.annotations))
        print("annotation descriptions:", sorted(set(raw.annotations.description))[:30])
        events, event_id = mne.events_from_annotations(raw)
        print("n_events:", len(events))
        print("event_id keys:", list(event_id)[:20])

        # Events from annotations (BNCI ErrP style)
        events, event_id = mne.events_from_annotations(raw)

        # Make each epoch variant from the same preprocessed raw
        for ep_name, ep_cfg in EPOCH_VARIANTS.items():
            print(f"  Epoching variant: {ep_name} (tmin={ep_cfg['tmin']}, tmax={ep_cfg['tmax']})")

            epochs = mne.Epochs( raw, events, event_id=event_id, tmin=ep_cfg["tmin"], tmax=ep_cfg["tmax"], baseline=ep_cfg["baseline"], preload=True, reject_by_annotation=True, verbose="ERROR")

            out_dir = EPOCHS_ROOT / ep_name / variant / sub / ses
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{base}-epo.fif"
            epochs.save(out_path, overwrite=True)
            print(f"    Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()