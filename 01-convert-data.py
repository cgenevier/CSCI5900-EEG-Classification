"""
BNCI/Matlab (.mat) -> MNE Raw -> FIF + BIDS EEG export (BrainVision)

Requirements:
  pip install mne mne-bids scipy numpy pandas pybv

Notes:
- Assumes MAT structure: mat["run"] is a cell-like array of run items.
- Each run item has: .eeg (samples x channels), .header with:
    Subject, Session, SampleRate, Label, EVENT.POS, EVENT.TYP

Export modes:
- EXPORT_MODE = "per_run": write one BIDS recording per MAT run (run-01, run-02, ...)
- EXPORT_MODE = "concatenate": concatenate MAT runs into one recording (single run-01)
"""

from __future__ import annotations

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
import mne

# -------------------------
# Config
# -------------------------
MATLAB_DIR = Path("datasets/bnci_horizon_2020_ErrP/original_matlab")

# MNE-native working / derivative files (segregated by subject/session)
OUT_FIF_DIR = Path("datasets/bnci_horizon_2020_ErrP/raw_fif")
OUT_BIDS_ROOT = Path("datasets/bnci_horizon_2020_ErrP/bids")

BIDS_TASK = "errp"

# Choose: "per_run" or "concatenate"
EXPORT_MODE = "per_run"

# For concatenate mode, which BIDS run number to use
CONCAT_BIDS_RUN = "01"

EVENT_ID = {
    "1": 1,
    "2": 2,
    "4": 4,
    "correct_5": 5,
    "error_6": 6,
    "8": 8,
    "error_9": 9,
    "correct_10": 10,
}

DEFAULT_CH_TYPE = "eeg"
ASSUME_MICROVOLTS = True # BNCI ErrP dataset is stored in microvolts


# -------------------------
# Filename parsing
# -------------------------
_SUB_SES_RE = re.compile(r"^Subject(?P<sub>\d+)_s(?P<ses>\d+)\.mat$", re.IGNORECASE)


def parse_subject_session_from_filename(mat_path: Path) -> tuple[str, str]:
    """
    Returns (subject, session) as zero-padded strings suitable for BIDS, e.g. ("01", "01").
    """
    m = _SUB_SES_RE.match(mat_path.name)
    if not m:
        raise ValueError(f"MAT filename does not match 'SubjectXX_sY.mat': {mat_path.name}")

    sub = int(m.group("sub"))
    ses = int(m.group("ses"))

    # Subject is typically 2-digit here; session we also pad to 2 for consistency
    return f"{sub:02d}", f"{ses:02d}"


# -------------------------
# Helpers (MAT unwrapping)
# -------------------------
def _unwrap(v):
    if isinstance(v, np.ndarray):
        v = np.array(v).squeeze()
        if v.dtype == object and v.size == 1:
            v = v.item()
        if isinstance(v, np.ndarray) and v.size == 1:
            v = v.item()
    return v


def _as_1d_array(v):
    v = _unwrap(v)
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return np.array(v).ravel()
    if isinstance(v, np.ndarray):
        return np.array(v).ravel()
    return np.array([v]).ravel()


def _matstruct_fields(ms):
    if hasattr(ms, "__dict__"):
        return [k for k in ms.__dict__.keys() if not k.startswith("_")]
    return [k for k in dir(ms) if not k.startswith("_")]


def _get_field_case_insensitive(ms, name: str):
    fields = _matstruct_fields(ms)
    lower_map = {f.lower(): f for f in fields}
    key = lower_map.get(name.lower())
    return getattr(ms, key) if key else None


def _to_str_list(v):
    arr = _as_1d_array(v)
    if arr is None:
        return None
    return [str(_unwrap(x)) for x in arr]


# -------------------------
# Extraction from BNCI-style MAT
# -------------------------
def extract_eeg(run_item) -> np.ndarray:
    eeg = np.array(run_item.eeg)
    if eeg.ndim != 2:
        raise ValueError(f"Expected 2D EEG, got shape {eeg.shape}")
    return eeg  # (samples, channels)


def extract_header(run_item) -> dict:
    hdr = run_item.header
    meta = {}

    if hasattr(hdr, "Subject"):
        meta["subject"] = str(_unwrap(hdr.Subject))
    if hasattr(hdr, "Session"):
        meta["session"] = str(_unwrap(hdr.Session))
    if hasattr(hdr, "SampleRate"):
        meta["sfreq"] = float(_unwrap(hdr.SampleRate))
    if hasattr(hdr, "Label"):
        labels = _to_str_list(hdr.Label)
        if labels:
            meta["ch_names"] = labels

    meta["header_fields_present"] = _matstruct_fields(hdr)
    return meta


def extract_events(run_item) -> pd.DataFrame:
    """
    Return DataFrame with sample_matlab (1-based), sample0 (0-based), code (int).
    """
    hdr = run_item.header
    event_ms = getattr(hdr, "EVENT", None)
    if event_ms is None:
        return pd.DataFrame(columns=["sample_matlab", "sample0", "code"])

    pos = _get_field_case_insensitive(event_ms, "POS")
    typ = _get_field_case_insensitive(event_ms, "TYP")

    if pos is None or typ is None:
        raise ValueError(f"EVENT missing POS or TYP. Fields: {_matstruct_fields(event_ms)}")

    pos = _as_1d_array(pos).astype(float)
    typ = _as_1d_array(typ)

    if len(pos) != len(typ):
        raise ValueError(f"POS/TYP length mismatch: {len(pos)} vs {len(typ)}")

    sample_matlab = np.round(pos).astype(int)
    sample0 = sample_matlab - 1

    codes = []
    for x in typ:
        x = _unwrap(x)
        codes.append(int(round(float(x))))

    return pd.DataFrame({"sample_matlab": sample_matlab, "sample0": sample0, "code": codes})


# -------------------------
# MAT loading / normalization
# -------------------------
def load_mat_runs(mat_path: Path):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "run" not in mat:
        raise KeyError(f"No 'run' in MAT. Keys: {[k for k in mat.keys() if not k.startswith('__')]}")

    run = mat["run"]
    if not hasattr(run, "shape"):
        run = np.array([run], dtype=object)

    # Make sure it's 1D iterable over runs
    run = np.array(run).reshape(-1)

    return run


# -------------------------
# Build standardized MNE objects
# -------------------------
def mat_to_mne_per_run(mat_path: Path):
    """
    Yield one (raw, events, events_df, meta, run_idx1) per BNCI run in the MAT.
    """
    runs = load_mat_runs(mat_path)
    n_runs = int(runs.shape[0])

    meta0 = extract_header(runs[0])
    sfreq = meta0.get("sfreq", None)
    if sfreq is None:
        raise ValueError("Could not read SampleRate (sfreq) from header.")

    eeg0 = extract_eeg(runs[0])
    n_ch = eeg0.shape[1]

    ch_names = meta0.get("ch_names", None)
    if not ch_names or len(ch_names) != n_ch:
        ch_names = [f"ch_{i}" for i in range(n_ch)]

    for i in range(n_runs):
        eeg = extract_eeg(runs[i])
        if eeg.shape[1] != n_ch:
            raise ValueError(f"Run {i} channel count mismatch: {eeg.shape[1]} vs {n_ch}")

        if ASSUME_MICROVOLTS:
            eeg = eeg * 1e-6  # µV -> V

        data = eeg.T
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=[DEFAULT_CH_TYPE] * n_ch)
        raw = mne.io.RawArray(data, info, verbose=False)

        events_df = extract_events(runs[i])
        events_df["mat_run_index0"] = i
        events_df["mat_run_index1"] = i + 1

        if len(events_df) > 0:
            events = np.column_stack(
                [
                    events_df["sample0"].astype(int).to_numpy(),
                    np.zeros(len(events_df), dtype=int),
                    events_df["code"].astype(int).to_numpy(),
                ]
            )
        else:
            events = np.zeros((0, 3), dtype=int)

        meta = {
            "source_mat": str(mat_path),
            "export_mode": "per_run",
            "n_runs_in_mat": n_runs,
            "mat_run_index0": int(i),
            "mat_run_index1": int(i + 1),
            "n_samples": int(eeg.shape[0]),
            "n_channels": int(n_ch),
            "sfreq": float(sfreq),
            "ch_names": ch_names,
            "header_fields_present": meta0.get("header_fields_present", []),
            "input_units": "µV" if ASSUME_MICROVOLTS else "V",
            "stored_units": "V",
            "assumed_microvolts_converted": bool(ASSUME_MICROVOLTS),
        }

        yield raw, events, events_df, meta, (i + 1)


def mat_to_mne_concatenated(mat_path: Path):
    """
    Return a single (raw, events, events_df, meta) created by concatenating all MAT runs.
    Events are shifted to global sample indices.
    """
    runs = load_mat_runs(mat_path)
    n_runs = int(runs.shape[0])

    meta0 = extract_header(runs[0])
    sfreq = meta0.get("sfreq", None)
    if sfreq is None:
        raise ValueError("Could not read SampleRate (sfreq) from header.")

    eeg0 = extract_eeg(runs[0])
    n_ch = eeg0.shape[1]

    ch_names = meta0.get("ch_names", None)
    if not ch_names or len(ch_names) != n_ch:
        ch_names = [f"ch_{i}" for i in range(n_ch)]

    eeg_runs = []
    samples_per_run = []
    for i in range(n_runs):
        eeg = extract_eeg(runs[i])
        if eeg.shape[1] != n_ch:
            raise ValueError(f"Run {i} channel count mismatch: {eeg.shape[1]} vs {n_ch}")
        if ASSUME_MICROVOLTS:
            eeg = eeg * 1e-6
        eeg_runs.append(eeg)
        samples_per_run.append(int(eeg.shape[0]))

    eeg_all = np.concatenate(eeg_runs, axis=0)
    data = eeg_all.T

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=[DEFAULT_CH_TYPE] * n_ch)
    raw = mne.io.RawArray(data, info, verbose=False)

    run_offsets = np.cumsum([0] + samples_per_run[:-1])

    all_events = []
    for i in range(n_runs):
        ev = extract_events(runs[i])
        if len(ev) == 0:
            continue
        ev["mat_run_index0"] = i
        ev["mat_run_index1"] = i + 1
        ev["sample_global0"] = run_offsets[i] + ev["sample0"]
        all_events.append(ev)

    events_df = (
        pd.concat(all_events, ignore_index=True)
        if all_events
        else pd.DataFrame(columns=["sample_matlab", "sample0", "code", "mat_run_index0", "mat_run_index1", "sample_global0"])
    )

    if len(events_df) > 0:
        events = np.column_stack(
            [
                events_df["sample_global0"].astype(int).to_numpy(),
                np.zeros(len(events_df), dtype=int),
                events_df["code"].astype(int).to_numpy(),
            ]
        )
    else:
        events = np.zeros((0, 3), dtype=int)

    meta = {
        "source_mat": str(mat_path),
        "export_mode": "concatenate",
        "n_runs_in_mat": int(n_runs),
        "samples_per_run": samples_per_run,
        "n_samples_total": int(sum(samples_per_run)),
        "n_channels": int(n_ch),
        "sfreq": float(sfreq),
        "ch_names": ch_names,
        "header_fields_present": meta0.get("header_fields_present", []),
        "input_units": "µV" if ASSUME_MICROVOLTS else "V",
        "stored_units": "V",
        "assumed_microvolts_converted": bool(ASSUME_MICROVOLTS),
    }

    return raw, events, events_df, meta


# -------------------------
# Save FIF + BIDS
# -------------------------
def save_fif(raw: mne.io.BaseRaw, out_root: Path, subject: str, session: str, task: str, run: str):
    out_dir = out_root / f"sub-{subject}" / f"ses-{session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fif_path = out_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_raw.fif"
    raw.save(fif_path, overwrite=True)
    return fif_path


def save_aux_files(out_root: Path, subject: str, session: str, task: str, run: str,
                   events_df: pd.DataFrame, meta: dict):
    aux_dir = out_root / f"sub-{subject}" / f"ses-{session}"
    aux_dir.mkdir(parents=True, exist_ok=True)

    events_csv = aux_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.csv"
    events_df.to_csv(events_csv, index=False)

    meta_json = aux_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_meta.json"
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    return events_csv, meta_json


def save_bids(raw: mne.io.BaseRaw, events: np.ndarray, out_root: Path,
              subject: str, session: str, task: str, run: str):
    from mne_bids import BIDSPath, write_raw_bids

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype="eeg",
        root=out_root,
    )

    write_raw_bids(
        raw,
        bids_path=bids_path,
        events=events if len(events) else None,
        event_id=EVENT_ID,
        overwrite=True,
        verbose=False,
        allow_preload=True,
        format="BrainVision",
    )
    return bids_path


# -------------------------
# Main
# -------------------------
def main():
    if EXPORT_MODE not in {"per_run", "concatenate"}:
        raise ValueError("EXPORT_MODE must be either 'per_run' or 'concatenate'")

    mat_files = sorted(MATLAB_DIR.glob("Subject*_s*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No MAT files found in: {MATLAB_DIR}")

    OUT_FIF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_BIDS_ROOT.mkdir(parents=True, exist_ok=True)

    for mat_path in mat_files:
        subject, session = parse_subject_session_from_filename(mat_path)
        print(f"\n=== Processing {mat_path.name} -> sub-{subject} ses-{session} (mode={EXPORT_MODE}) ===")

        if EXPORT_MODE == "per_run":
            for raw, events, events_df, meta, run_idx1 in mat_to_mne_per_run(mat_path):
                bids_run = f"{run_idx1:02d}"  # run-01, run-02, ...

                print(f"--- MAT run {run_idx1} -> BIDS run {bids_run} ---")

                fif_path = save_fif(raw, OUT_FIF_DIR, subject, session, BIDS_TASK, bids_run)
                events_csv, meta_json = save_aux_files(
                    OUT_FIF_DIR, subject, session, BIDS_TASK, bids_run, events_df, meta
                )
                bids_path = save_bids(raw, events, OUT_BIDS_ROOT, subject, session, BIDS_TASK, bids_run)

                print(f"Saved FIF:  {fif_path}")
                print(f"Saved CSV:  {events_csv}")
                print(f"Saved META: {meta_json}")
                print(f"Saved BIDS: {bids_path.directory}")
                print(f"Sanity: {events.shape[0]} events, {raw.n_times} samples, {raw.info['nchan']} ch")

        else:  # concatenate
            raw, events, events_df, meta = mat_to_mne_concatenated(mat_path)
            bids_run = CONCAT_BIDS_RUN

            fif_path = save_fif(raw, OUT_FIF_DIR, subject, session, BIDS_TASK, bids_run)
            events_csv, meta_json = save_aux_files(
                OUT_FIF_DIR, subject, session, BIDS_TASK, bids_run, events_df, meta
            )
            bids_path = save_bids(raw, events, OUT_BIDS_ROOT, subject, session, BIDS_TASK, bids_run)

            print(f"Saved FIF:  {fif_path}")
            print(f"Saved CSV:  {events_csv}")
            print(f"Saved META: {meta_json}")
            print(f"Saved BIDS: {bids_path.directory}")
            print(f"Sanity: {events.shape[0]} events, {raw.n_times} samples, {raw.info['nchan']} ch")


if __name__ == "__main__":
    main()