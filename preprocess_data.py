#!/usr/bin/env python3
"""Unified preprocessing entrypoint for DREAMT and Sleep-EDF into chunked NPZ files.

This script mirrors dataset-specific sanitization from existing code:
- Sleep-EDF: follows preprocess_sleep_edf.py stage mapping, wake trimming, resampling,
  and SpectrogramTransformer preprocessing.
- DREAMT: follows src/common/datasets/dreamt.py label canonicalization and per-segment
  spectrogram image conversion.

Outputs are written in chunked triplets:
1) preprocessed_spectrograms_*.npz
2) preprocessed_timeseries_*.npz
3) preprocessed_non_signal_*.npz

Each NPZ contains a metadata JSON string (`metadata_json`) and typed arrays keyed by
metadata references. Load with numpy as usual:
    data = np.load(path)
    meta = json.loads(str(data["metadata_json"]))
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm



# -------------------------------
# Shared / serialization helpers
# -------------------------------


@dataclass
class ProcessedRecord:
    record_id: str
    subject_id: str
    spectrograms: Dict[str, np.ndarray]
    timeseries: Dict[str, np.ndarray]
    non_signal: Dict[str, np.ndarray]


def infer_subject_id(dataset_tag: str, record_id: str) -> str:
    rid = str(record_id).strip()
    if dataset_tag == "dreamt":
        m = re.search(r"S\d+", rid, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()
        return rid.split("_")[0] if "_" in rid else rid

    if dataset_tag == "sleep_edf":
        # SC4xxnE0 -> subject is SC4xx (night index removed)
        m = re.match(r"^(SC4\d{2})[12]E0$", rid, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.match(r"^(SC\d{2})", rid, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return rid

    return rid


def _safe_key(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "field"


def _to_np(v: np.ndarray | Iterable | str | int | float) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (str, int, float)):
        return np.asarray(v)
    return np.asarray(list(v))


def _write_kind_npz(
    output_dir: Path,
    kind: str,
    dataset_tag: str,
    file_tag: str,
    records: List[ProcessedRecord],
    selector,
) -> Path:
    payload: Dict[str, np.ndarray] = {}
    metadata = {
        "dataset": dataset_tag,
        "kind": kind,
        "file_tag": file_tag,
        "num_records": int(len(records)),
        "records": [],
    }

    for rec_i, rec in enumerate(records):
        bundle = selector(rec)
        field_to_key: Dict[str, str] = {}
        field_shapes: Dict[str, List[int]] = {}
        field_dtypes: Dict[str, str] = {}
        for field, arr in bundle.items():
            key = f"r{rec_i}_{_safe_key(field)}"
            arr_np = _to_np(arr)
            payload[key] = arr_np
            field_to_key[field] = key
            field_shapes[field] = list(arr_np.shape)
            field_dtypes[field] = str(arr_np.dtype)

        metadata["records"].append(
            {
                "record_id": rec.record_id,
                "subject_id": rec.subject_id,
                "fields": field_to_key,
                "shapes": field_shapes,
                "dtypes": field_dtypes,
            }
        )

    payload["metadata_json"] = np.asarray(json.dumps(metadata), dtype=np.str_)
    filename = f"preprocessed_{kind}_{dataset_tag}_{file_tag}.npz"
    out_path = output_dir / filename
    np.savez_compressed(out_path, **payload)
    return out_path


def _write_subject_triplet(
    output_dir: Path,
    dataset_tag: str,
    subject_id: str,
    records: List[ProcessedRecord],
) -> Tuple[Path, Path, Path]:
    file_tag = f"subject_{_safe_key(subject_id)}"
    spec_path = _write_kind_npz(
        output_dir=output_dir,
        kind="spectrograms",
        dataset_tag=dataset_tag,
        file_tag=file_tag,
        records=records,
        selector=lambda r: r.spectrograms,
    )
    ts_path = _write_kind_npz(
        output_dir=output_dir,
        kind="timeseries",
        dataset_tag=dataset_tag,
        file_tag=file_tag,
        records=records,
        selector=lambda r: r.timeseries,
    )
    non_sig_path = _write_kind_npz(
        output_dir=output_dir,
        kind="non_signal",
        dataset_tag=dataset_tag,
        file_tag=file_tag,
        records=records,
        selector=lambda r: r.non_signal,
    )
    return spec_path, ts_path, non_sig_path


def _subject_triplet_paths(output_dir: Path, dataset_tag: str, subject_id: str) -> Tuple[Path, Path, Path]:
    tag = f"subject_{_safe_key(subject_id)}"
    return (
        output_dir / f"preprocessed_spectrograms_{dataset_tag}_{tag}.npz",
        output_dir / f"preprocessed_timeseries_{dataset_tag}_{tag}.npz",
        output_dir / f"preprocessed_non_signal_{dataset_tag}_{tag}.npz",
    )


def _subject_triplet_exists(output_dir: Path, dataset_tag: str, subject_id: str) -> bool:
    a, b, c = _subject_triplet_paths(output_dir, dataset_tag, subject_id)
    return a.exists() and b.exists() and c.exists()

def normalize_signal_1d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-mean, unit-variance normalize a 1D tensor with numerical safety."""
    if x.dim() != 1:
        raise ValueError('Expected a 1D tensor for normalization')
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x - mean) / (std + eps)

class SpectrogramNormalizer:
    """Maintains running stats to normalize spectrograms consistently across calls."""

    def __init__(self, momentum: float = 0.1, eps: float = 1e-5):
        self.momentum = momentum
        self.eps = eps
        self._running_mean = None
        self._running_var = None
        self._count = 0

    def _update_stats(self, mean: torch.Tensor, var: torch.Tensor, device: torch.device):
        if self._running_mean is None or self._running_var is None:
            self._running_mean = mean.to(device)
            self._running_var = var.to(device)
        else:
            self._running_mean = (1.0 - self.momentum) * self._running_mean + self.momentum * mean.to(device)
            self._running_var = (1.0 - self.momentum) * self._running_var + self.momentum * var.to(device)
        self._count += 1

    def normalize(self, spec: torch.Tensor) -> torch.Tensor:
        spec_f = spec.float()
        mean = spec_f.mean()
        var = spec_f.var(unbiased=False)
        self._update_stats(mean, var, spec_f.device)
        std = torch.sqrt(self._running_var + self.eps)
        return (spec_f - self._running_mean) / std

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        return self.normalize(spec)

# load file from csv and make spectrograms
class SpectrogramTransformer:
    def __init__(self, n_fft=512, hop_length=128, win_length=512, sr=256, db_min=-80.0):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sr = sr
        self.db_min = db_min
        # running normalizer keeps spectrograms on a shared scale across calls
        self.spec_normalizer = SpectrogramNormalizer()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a 1D tensor of audio samples into a normalized log-magnitude spectrogram.
        
        Args:
            x (torch.Tensor): 1D tensor of audio samples.
        
        Returns:
            torch.Tensor: 2D tensor representing the normalized log-magnitude spectrogram.
        """
        # ensure input is 1D
        if x.dim() != 1:
            raise ValueError("Input tensor must be 1-dimensional")

        # normalize raw signal before STFT
        x = normalize_signal_1d(x)

        # ensure window is on same device
        try:
            window = torch.hamming_window(self.win_length, device=x.device)
        except Exception:
            # fallback to CPU window
            window = torch.hamming_window(self.win_length)

        # compute complex STFT (center=True for common behaviour)
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=window, center=True, return_complex=True)

        # compute magnitude
        magnitude = torch.abs(stft)

        # convert to decibels
        db_spec = 20.0 * torch.log10(torch.clamp(magnitude, min=1e-10))

        # normalize to [0, 1]
        # db_spec = (db_spec - self.db_min) / -self.db_min

        # final safety
        db_spec = torch.nan_to_num(db_spec, nan=0.0, posinf=0.0, neginf=0.0)
        db_spec = self.spec_normalizer(db_spec)
        # print(db_spec.shape)
        return db_spec


# -------------------------------
# Sleep-EDF preprocessing module
# -------------------------------

STAGE_MAP = {
    "sleep stage w": "W",
    "sleep stage 1": "N1",
    "sleep stage 2": "N2",
    "sleep stage 3": "N3",
    "sleep stage 4": "N3",
    "Sleep stage R": "R",
    "sleep stage r": "R",
    "wake": "W",
    "sleep stage m": "Missing",
    "movement time": "Missing",
    "sleep stage ?": "Missing",
}

BASE_STAGE_CLASSES = {"W", "N1", "N2", "N3", "R"}

CANONICAL_MODALITIES = {
    "EEG1": ["EEG_FpzCz"],
    "EEG2": ["EEG_PzOz"],
    "EOG": ["EOG_horizontal"],
    "EMG": ["EMG_submental"],
    "Resp": ["Resp_oro-nasal"],
    "Temp": ["Temp_rectal"],
    "EEG_FpzCz": ["EEG1"],
    "EEG_PzOz": ["EEG2"],
}

SLEEP_EDF20_SC_PSG_RE = re.compile(r"^SC4(0[0-9]|1[0-9])[12]E0-PSG\.edf$")


def is_sleep_edf20_sc_psg(path: Path) -> bool:
    return bool(SLEEP_EDF20_SC_PSG_RE.match(path.name))


def normalize_label(label: str) -> str:
    cleaned = label.strip().upper()
    for ch in ["-", "_", "/", "(", ")"]:
        cleaned = cleaned.replace(ch, " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def standardize_stage(label: str) -> str:
    key = label.strip().lower()
    if key in STAGE_MAP:
        return STAGE_MAP[key]
    tqdm.write(f"Warning: Unrecognized sleep stage label '{label}'; mapping to 'Missing'")
    return "Missing"


def resample_signal(signal: np.ndarray, orig_sr: float, target_sr: int) -> np.ndarray:
    if not orig_sr or orig_sr <= 0:
        raise ValueError(f"Invalid original sampling rate: {orig_sr}")
    if math.isclose(orig_sr, target_sr, rel_tol=1e-3):
        return signal.astype(np.float32, copy=False)
    duration = len(signal) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    orig_times = np.linspace(0.0, duration, num=len(signal), endpoint=False, dtype=np.float64)
    target_times = np.linspace(0.0, duration, num=target_len, endpoint=False, dtype=np.float64)
    resampled = np.interp(target_times, orig_times, signal.astype(np.float64))
    return resampled.astype(np.float32)


def find_hypnogram_for_psg(psg_path: Path, hyp_dir: Path) -> Optional[Path]:
    record_id = psg_path.stem.replace("-PSG", "")
    exact = hyp_dir / f"{record_id}-Hypnogram.edf"
    if exact.exists():
        return exact
    if record_id.endswith("0"):
        stem_prefix = record_id[:-1]
        candidates = sorted(hyp_dir.glob(f"{stem_prefix}?-Hypnogram.edf"))
        if candidates:
            return candidates[0]
    candidates = sorted(hyp_dir.glob(f"{record_id[:7]}*-Hypnogram.edf"))
    if candidates:
        return candidates[0]
    return None


def read_hypnogram_annotations(hyp_path: Path) -> List[Tuple[float, float, str]]:
    from pyedflib import highlevel

    if not hyp_path.exists():
        return []
    try:
        annotations = highlevel.read_annotations(str(hyp_path))
        return [(float(a[0]), float(a[1]), str(a[2])) for a in annotations]
    except Exception:
        pass
    try:
        import pyedflib

        with pyedflib.EdfReader(str(hyp_path)) as reader:
            onsets, durations, descriptions = reader.readAnnotations()
        return [(float(o), float(d), str(desc)) for o, d, desc in zip(onsets, durations, descriptions)]
    except Exception:
        return []


def build_stage_series(
    annotations: List[Tuple[float, float, str]], total_samples: int, target_sr: int
) -> np.ndarray:
    stage_series = np.full(total_samples, "Missing", dtype=object)
    for onset, duration, description in annotations:
        if duration <= 0:
            continue
        label = standardize_stage(description)
        start_idx = int(onset * target_sr)
        end_idx = int((onset + duration) * target_sr)
        if end_idx <= 0:
            continue
        start_idx = max(0, start_idx)
        end_idx = min(total_samples, end_idx)
        if start_idx >= end_idx:
            continue
        stage_series[start_idx:end_idx] = label
    return stage_series


def build_requested_modalities(raw_modalities: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    alias_map = {}
    name_map = {}
    for canonical, aliases in CANONICAL_MODALITIES.items():
        norm_aliases = {normalize_label(a) for a in aliases}
        alias_map[canonical] = norm_aliases
        for alias in norm_aliases:
            name_map[alias] = canonical
    ordered: List[str] = []
    matchers: Dict[str, List[str]] = {}
    for mod in raw_modalities:
        norm = normalize_label(mod)
        canonical = name_map.get(norm, mod)
        if canonical in matchers:
            continue
        aliases = set()
        if canonical in alias_map:
            aliases |= alias_map[canonical]
        aliases.add(norm)
        matchers[canonical] = sorted(aliases)
        ordered.append(canonical)
    return ordered, matchers


def make_unique_labels(labels: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    unique: List[str] = []
    for label in labels:
        count = counts.get(label, 0) + 1
        counts[label] = count
        if count == 1:
            unique.append(label)
        else:
            unique.append(f"{label}_{count}")
    return unique


def load_modalities(
    edf_path: Path, requested_modalities: List[str], target_sr: int, include_all: bool
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    from pyedflib import highlevel

    signals, signal_headers, header = highlevel.read_edf(str(edf_path), digital=False)
    if include_all:
        raw_labels = [hdr["label"].strip() for hdr in signal_headers]
        ordered_modalities = make_unique_labels(raw_labels)
        matchers = {}
    else:
        ordered_modalities, matchers = build_requested_modalities(requested_modalities)

    data: Dict[str, np.ndarray] = {}
    for idx, hdr in enumerate(signal_headers):
        raw_label = hdr["label"].strip()
        norm_label = normalize_label(raw_label)
        matched = None
        if include_all:
            matched = ordered_modalities[idx]
        else:
            for canonical in ordered_modalities:
                if norm_label in matchers.get(canonical, []):
                    matched = canonical
                    break
        if matched is None:
            continue
        if matched in data:
            continue
        sr = hdr.get("sample_rate") or hdr.get("sample_frequency")
        if sr is None:
            sr = header.get("record_frequency")
        if sr is None:
            raise ValueError(f"Could not determine sampling rate for {raw_label} in {edf_path.name}")
        signal = np.asarray(signals[idx], dtype=np.float32)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        data[matched] = resample_signal(signal, float(sr), target_sr)

    return data, ordered_modalities


def compute_spectrogram_segments(
    signal: np.ndarray, processor, segment_samples: int
) -> np.ndarray:
    import torch

    usable = (len(signal) // segment_samples) * segment_samples
    if usable == 0:
        return np.empty((0, 0), dtype=np.float32)
    tensor = torch.tensor(signal[:usable], dtype=torch.float32)
    segments = tensor.view(-1, segment_samples)
    specs = [processor.preprocess(seg) for seg in segments]
    flattened = [spec.detach().cpu().numpy().flatten() for spec in specs]
    if not flattened:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(flattened, axis=0).astype(np.float32, copy=False)


def trim_wake_segments(segment_labels: List[str], max_wake_segments: int) -> np.ndarray:
    labels = np.asarray(segment_labels, dtype=object)
    base_mask = np.array([lbl in BASE_STAGE_CLASSES for lbl in labels], dtype=bool)
    if not np.any(base_mask):
        return base_mask
    non_w_mask = np.array([lbl in BASE_STAGE_CLASSES and lbl != "W" for lbl in labels], dtype=bool)
    if not np.any(non_w_mask):
        idx = np.where(base_mask)[0]
        if idx.size > max_wake_segments:
            trimmed = np.zeros_like(base_mask)
            trimmed[idx[:max_wake_segments]] = True
            return trimmed
        return base_mask

    first_sleep = int(np.where(non_w_mask)[0][0])
    last_sleep = int(np.where(non_w_mask)[0][-1])
    trimmed = base_mask.copy()

    if first_sleep > 0:
        pre_idx = np.where(labels[:first_sleep] == "W")[0]
        if pre_idx.size > max_wake_segments:
            drop = pre_idx[: -max_wake_segments]
            trimmed[drop] = False

    if last_sleep < len(labels) - 1:
        post_idx = np.where(labels[last_sleep + 1 :] == "W")[0] + last_sleep + 1
        if post_idx.size > max_wake_segments:
            drop = post_idx[max_wake_segments:]
            trimmed[drop] = False

    return trimmed


def process_sleep_edf_record(
    psg_path: Path,
    hyp_path: Path,
    processor,
    modalities_arg: str,
    target_sr: int,
    segment_length: int,
    max_wake_segments: int,
) -> Optional[ProcessedRecord]:
    raw_modalities = [m.strip() for m in modalities_arg.split(",") if m.strip()]
    include_all = any(m.upper() in {"ALL", "*"} for m in raw_modalities) or not raw_modalities
    if include_all:
        raw_modalities = []

    modality_data, ordered_modalities = load_modalities(
        psg_path, raw_modalities, target_sr, include_all
    )
    if not modality_data:
        tqdm.write(f"No requested modalities found in {psg_path.name}; skipping")
        return None

    min_len = min(len(arr) for arr in modality_data.values())
    segment_samples = target_sr * segment_length
    usable_samples = (min_len // segment_samples) * segment_samples
    if usable_samples == 0:
        tqdm.write(f"Record {psg_path.name} shorter than one {segment_length}s window; skipping")
        return None

    for mod in modality_data:
        sig = modality_data[mod][:usable_samples]
        modality_data[mod] = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    annotations = read_hypnogram_annotations(hyp_path)
    stage_series = build_stage_series(annotations, usable_samples, target_sr)
    timestamps = np.arange(usable_samples, dtype=np.float64) / float(target_sr)

    num_segments = usable_samples // segment_samples
    stage_segments = stage_series.reshape(num_segments, segment_samples)
    segment_labels: List[str] = []
    for seg in stage_segments:
        labels, counts = np.unique(seg, return_counts=True)
        segment_labels.append(str(labels[int(np.argmax(counts))]))

    keep_mask = trim_wake_segments(segment_labels, max_wake_segments)
    if not np.any(keep_mask):
        tqdm.write(f"All segments filtered (non-base classes) for {psg_path.name}; skipping")
        return None

    keep_timepoints = np.repeat(keep_mask, segment_samples)[:usable_samples]
    stage_series_kept = stage_series[keep_timepoints].astype("U16")
    timestamps_kept = timestamps[keep_timepoints].astype(np.float32)

    spectrograms: Dict[str, np.ndarray] = {}
    timeseries: Dict[str, np.ndarray] = {}

    for mod in ordered_modalities:
        signal = modality_data.get(mod)
        if signal is None:
            continue

        ts_segments = signal.reshape(num_segments, segment_samples)
        ts_segments = ts_segments[keep_mask].astype(np.float32, copy=False)
        if ts_segments.size == 0:
            tqdm.write(f"All time-series segments filtered for {psg_path.name} after modality {mod}")
            return None

        spec_segments = compute_spectrogram_segments(signal, processor, segment_samples)
        if spec_segments.shape[0] != num_segments:
            n = min(spec_segments.shape[0], num_segments)
            spec_segments = spec_segments[:n]
            ts_segments = ts_segments[:n]

        spec_segments = spec_segments[keep_mask[: spec_segments.shape[0]]]
        if spec_segments.size == 0:
            tqdm.write(f"All spectrogram segments filtered for {psg_path.name} after modality {mod}")
            return None

        spectrograms[mod] = spec_segments
        timeseries[mod] = ts_segments

    if not spectrograms or not timeseries:
        return None

    record_id = psg_path.stem.replace("-PSG", "")
    non_signal = {
        "TIMESTAMP": timestamps_kept,
        "Sleep_Stage": stage_series_kept,
        "segment_labels": np.asarray([segment_labels[i] for i, k in enumerate(keep_mask) if k], dtype="U16"),
        "subject_id": np.asarray([infer_subject_id("sleep_edf", record_id)] * len(stage_series_kept), dtype="U16"),
    }

    return ProcessedRecord(
        record_id=record_id,
        subject_id=infer_subject_id("sleep_edf", record_id),
        spectrograms=spectrograms,
        timeseries=timeseries,
        non_signal=non_signal,
    )


def run_sleep_edf(args: argparse.Namespace, output_dir: Path) -> int:

    processor = SpectrogramTransformer(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        sr=args.target_sr,
        db_min=-80.0,
    )

    psg_files = sorted(args.sleep_edf_psg_dir.glob("*-PSG.edf"))
    if args.sleep_edf20_only:
        psg_files = [p for p in psg_files if is_sleep_edf20_sc_psg(p)]
        tqdm.write(
            f"Sleep-EDF-20 filter enabled: {len(psg_files)} PSG file(s) matched in {args.sleep_edf_psg_dir}"
        )

    if args.limit_files > 0:
        psg_files = psg_files[: args.limit_files]

    if not psg_files:
        tqdm.write(f"No PSG EDF files found in {args.sleep_edf_psg_dir}")
        return 0

    dataset_tag = "sleep_edf"
    subject_to_records: Dict[str, List[ProcessedRecord]] = defaultdict(list)
    skipped_subjects: set[str] = set()
    processed = 0

    for psg_path in tqdm(psg_files, desc="Sleep-EDF records"):
        record_id = psg_path.stem.replace("-PSG", "")
        subject_id = infer_subject_id(dataset_tag, record_id)
        if not args.overwrite and _subject_triplet_exists(output_dir, dataset_tag, subject_id):
            if subject_id not in skipped_subjects:
                tqdm.write(f"Outputs exist for Sleep-EDF subject {subject_id}; skipping")
                skipped_subjects.add(subject_id)
            continue

        hyp_path = find_hypnogram_for_psg(psg_path, args.sleep_edf_hyp_dir)
        if hyp_path is None or not hyp_path.exists():
            tqdm.write(f"Missing hypnogram for {psg_path.name}; skipping")
            continue

        try:
            rec = process_sleep_edf_record(
                psg_path=psg_path,
                hyp_path=hyp_path,
                processor=processor,
                modalities_arg=args.sleep_edf_modalities,
                target_sr=args.target_sr,
                segment_length=args.segment_length,
                max_wake_segments=args.max_wake_segments,
            )
        except Exception as exc:
            tqdm.write(f"Failed to process {psg_path.name}: {exc}")
            continue

        if rec is None:
            continue

        subject_to_records[rec.subject_id].append(rec)
        processed += 1

    for subject_id, records in sorted(subject_to_records.items()):
        spec_p, ts_p, non_p = _write_subject_triplet(
            output_dir=output_dir,
            dataset_tag=dataset_tag,
            subject_id=subject_id,
            records=records,
        )
        tqdm.write(
            f"Wrote subject {subject_id}: {spec_p.name}, {ts_p.name}, {non_p.name}"
        )

    return processed


# -------------------------------
# DREAMT preprocessing module
# -------------------------------

TOKEN_TO_MODALITY = {
    "F": ("flow", ["FLOW", "flow"]),
    "E": ("ecg", ["ECG", "ecg"]),
    "C": ("chin", ["CHIN", "chin", "EMG", "emg", "EMG_submental"]),
    "A": ("bvp", ["BVP", "bvp"]),
    "B": ("acc_x", ["ACC_X", "acc_x"]),
    "D": ("acc_y", ["ACC_Y", "acc_y"]),
    "G": ("acc_z", ["ACC_Z", "acc_z"]),
    "H": ("eda", ["EDA", "eda"]),
    "I": ("temp", ["TEMP", "temp"]),
    "J": ("hr", ["HR", "hr"]),
    "K": ("ibi", ["IBI", "ibi"]),
    "L": ("c4_m1", ["C4-M1", "C4_M1", "c4-m1", "c4_m1"]),
    "M": ("f4_m1", ["F4-M1", "F4_M1", "f4-m1", "f4_m1"]),
    "N": ("o2_m1", ["O2-M1", "O2_M1", "o2-m1", "o2_m1"]),
    "O": ("t3_cz", ["T3-CZ", "T3_CZ", "T3 - CZ", "t3-cz", "t3_cz"]),
    "P": ("cz_t4", ["CZ-T4", "CZ_T4", "CZ - T4", "cz-t4", "cz_t4"]),
    "Q": ("e1", ["E1", "e1"]),
    "R": ("e2", ["E2", "e2"]),
    "S": ("ptaf", ["PTAF", "ptaf"]),
    "T": ("thorax", ["THORAX", "thorax"]),
    "U": ("abdomen", ["ABDOMEN", "abdomen"]),
    "V": ("snore", ["SNORE", "snore"]),
    "W": ("lat", ["LAT", "lat"]),
    "X": ("rat", ["RAT", "rat"]),
    "Y": ("sao2", ["SAO2", "sao2", "SpO2", "SPO2", "spo2"]),
}


DREAMT_LABEL_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}


def _normalize_name(name: str) -> str:
    return str(name).strip().upper().replace("-", "_").replace(" ", "_")


def _canonical_label(raw_label: object) -> Optional[str]:
    label = str(raw_label).strip().upper()
    if label in {"R", "REM"}:
        return "REM"
    if label in {"W", "N1", "N2", "N3", "REM"}:
        return label
    return None


def _label_to_idx(raw_label: object) -> int:
    canon = _canonical_label(raw_label)
    if canon is None:
        return -1
    return DREAMT_LABEL_MAP.get(canon, -1)


def _spec_target_shape(sr: int, segment_seconds: int, n_fft: int, hop_length: int) -> Tuple[int, int]:
    segment_samples = sr * segment_seconds
    pad = n_fft // 2
    freq_bins = n_fft // 2 + 1
    time_frames = 1 + max(0, (segment_samples + 2 * pad - n_fft) // max(1, hop_length))
    return freq_bins, time_frames


def _vector_to_rgb_spec(vector: np.ndarray, freq_bins: int, time_frames: int) -> np.ndarray:
    target_size = freq_bins * time_frames
    flat = np.zeros(target_size, dtype=np.float32)
    fill_size = min(target_size, vector.shape[0])
    if fill_size > 0:
        flat[:fill_size] = vector[:fill_size]
    spec = flat.reshape(freq_bins, time_frames)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(np.min(spec))
    max_val = float(np.max(spec))
    if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:
        spec = (spec - min_val) / (max_val - min_val + 1e-6)
    else:
        spec[:] = 0.0
    img = (spec * 255.0).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def _read_dreamt_pairs(data_dir: Path) -> List[Tuple[Path, Path, str]]:
    pre_csvs = sorted(
        [
            p
            for p in data_dir.glob("preprocessed_*.csv")
            if not p.name.startswith("preprocessed_non_signal_")
        ]
    )
    pairs: List[Tuple[Path, Path, str]] = []
    for p in pre_csvs:
        base = p.name.replace("preprocessed_", "", 1)
        non_signal = data_dir / f"preprocessed_non_signal_{base}"
        if non_signal.exists():
            record_id = base.replace(".csv", "")
            pairs.append((p, non_signal, record_id))
    return pairs


def _read_dreamt_raw_csvs(dreamt_root: Path) -> List[Path]:
    raw_dir = dreamt_root / "data_100Hz" if (dreamt_root / "data_100Hz").exists() else dreamt_root
    return sorted([p for p in raw_dir.glob("*.csv") if p.is_file()])


def _sanitize_dreamt_rows(person_file: pd.DataFrame, sr: int, segment_seconds: int) -> pd.DataFrame:
    if "TIMESTAMP" in person_file.columns:
        ts_match = person_file.index[person_file["TIMESTAMP"] == 1.0]
        if len(ts_match) > 0:
            person_file = person_file.iloc[int(ts_match[0]) :, :]

    if "Sleep_Stage" not in person_file.columns:
        return pd.DataFrame(columns=person_file.columns)

    series = person_file["Sleep_Stage"].to_numpy()
    invalid_set = {"P", "Missing"}
    invalid_indices = {i for i, v in enumerate(series) if str(v) in invalid_set}

    win = sr * segment_seconds
    kept_chunks = []
    i = 0
    total = len(series)
    while i < total:
        end = i + win
        if end <= total and i not in invalid_indices and all(x not in invalid_indices for x in range(i, end)):
            kept_chunks.append(person_file.iloc[i:end])
            i += win
        else:
            i += 1

    if not kept_chunks:
        return pd.DataFrame(columns=person_file.columns)

    return pd.concat(kept_chunks, axis=0, ignore_index=True)


def _resolve_tokens_from_arg(modality_arg: str, signal_columns: List[str]) -> List[str]:
    arg = str(modality_arg).strip()
    if not arg:
        raise ValueError("--dreamt_modalities cannot be empty")

    available = {_normalize_name(c): c for c in signal_columns}
    name_to_token: Dict[str, str] = {}
    for token, (canonical, candidates) in TOKEN_TO_MODALITY.items():
        name_to_token[_normalize_name(token)] = token
        name_to_token[_normalize_name(canonical)] = token
        for cand in candidates:
            name_to_token[_normalize_name(cand)] = token

    if arg.upper() in {"ALL", "*"}:
        resolved = []
        for token, (_, candidates) in TOKEN_TO_MODALITY.items():
            if any(_normalize_name(c) in available for c in candidates):
                resolved.append(token)
        if not resolved:
            raise ValueError("No known DREAMT modality columns found for ALL")
        return resolved

    if "," in arg:
        parts = [p.strip() for p in arg.split(",") if p.strip()]
    else:
        parts = list(arg)

    resolved = []
    for part in parts:
        token = name_to_token.get(_normalize_name(part))
        if token is None:
            raise ValueError(f"Unknown DREAMT modality selector: {part}")
        if token not in resolved:
            resolved.append(token)

    if not resolved:
        raise ValueError("No DREAMT modalities resolved from --dreamt_modalities")
    return resolved


def _resolve_dreamt_columns(signal_df: pd.DataFrame, modality_tokens: List[str]) -> Optional[Dict[str, str]]:
    normalized_to_raw = {_normalize_name(c): c for c in signal_df.columns}
    resolved: Dict[str, str] = {}
    for token in modality_tokens:
        _, candidates = TOKEN_TO_MODALITY[token]
        found = None
        for cand in candidates:
            norm = _normalize_name(cand)
            if norm in normalized_to_raw:
                found = normalized_to_raw[norm]
                break
        if found is None:
            return None
        resolved[token] = found
    return resolved


def _extract_segment_labels(non_signal_df: pd.DataFrame, sr: int, segment_seconds: int) -> Optional[np.ndarray]:
    if "Sleep_Stage" not in non_signal_df.columns:
        return None
    labels = non_signal_df["Sleep_Stage"].to_numpy()
    labels = np.asarray([_label_to_idx(v) for v in labels], dtype=np.int64)
    win = sr * segment_seconds
    usable = (len(labels) // win) * win
    if usable == 0:
        return None
    labels = labels[:usable].reshape(-1, win)

    seg_labels = []
    for seg in labels:
        valid = seg[seg >= 0]
        if valid.size == 0:
            seg_labels.append(-1)
        else:
            seg_labels.append(int(np.bincount(valid).argmax()))
    return np.asarray(seg_labels, dtype=np.int64)


def process_dreamt_record(
    signal_path: Path,
    record_id: str,
    modality_tokens: List[str],
    sr: int,
    segment_seconds: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> Optional[ProcessedRecord]:
    import torch

    person_file = pd.read_csv(signal_path)
    person_file = _sanitize_dreamt_rows(person_file, sr=sr, segment_seconds=segment_seconds)
    if len(person_file) == 0:
        return None

    col_map = _resolve_dreamt_columns(person_file, modality_tokens)
    if col_map is None:
        return None

    win = sr * segment_seconds
    if len(person_file) < win:
        return None

    n_segments = len(person_file) // win
    if n_segments <= 0:
        return None
    usable_rows = n_segments * win
    person_file = person_file.iloc[:usable_rows].copy()

    spectrograms: Dict[str, np.ndarray] = {}
    timeseries: Dict[str, np.ndarray] = {}
    spec_processor = SpectrogramTransformer(sr=sr * 4)
    spec_processor.n_fft = n_fft
    spec_processor.hop_length = hop_length
    spec_processor.win_length = win_length

    for token in modality_tokens:
        modality_name = TOKEN_TO_MODALITY[token][0]
        col = col_map[token]
        stream = person_file[col].to_numpy(dtype=np.float32).reshape(-1)
        stream = np.nan_to_num(stream, nan=0.0, posinf=0.0, neginf=0.0)
        if len(stream) < usable_rows:
            return None
        stream = stream[:usable_rows]

        ts_chunks = stream.reshape(n_segments, win).astype(np.float32, copy=False)
        timeseries[modality_name] = ts_chunks

        data = torch.tensor(stream, dtype=torch.float32)
        segment_tensors = data.split(win)
        specs = [spec_processor.preprocess(seg) for seg in segment_tensors]
        if not specs:
            return None
        flattened = np.stack([s.detach().cpu().numpy().flatten() for s in specs], axis=0).astype(np.float32)
        spectrograms[modality_name] = flattened

    non_signal_cols = [
        c
        for c in [
            "TIMESTAMP",
            "Sleep_Stage",
            "Mixed_Apnea",
            "Obstructive_Apnea",
            "Central_Apnea",
            "Hypopnea",
        ]
        if c in person_file.columns
    ]

    non_signal: Dict[str, np.ndarray] = {}
    for col in non_signal_cols:
        arr = person_file[col].to_numpy()
        if col == "Sleep_Stage":
            arr = np.asarray([_canonical_label(v) or "Missing" for v in arr], dtype="U16")
        elif np.issubdtype(np.asarray(arr).dtype, np.number):
            arr = np.asarray(arr, dtype=np.float32)
        else:
            arr = np.asarray(arr, dtype="U32")
        non_signal[col] = arr

    seg_labels = _extract_segment_labels(person_file.loc[:, ["Sleep_Stage"]], sr, segment_seconds)
    if seg_labels is None:
        return None
    non_signal["segment_label_idx"] = seg_labels.astype(np.int64, copy=False)
    subject_id = infer_subject_id("dreamt", record_id)
    non_signal["subject_id"] = np.asarray([subject_id] * len(person_file), dtype="U16")

    return ProcessedRecord(
        record_id=record_id,
        subject_id=subject_id,
        spectrograms=spectrograms,
        timeseries=timeseries,
        non_signal=non_signal,
    )


def run_dreamt(args: argparse.Namespace, output_dir: Path) -> int:
    raw_files = _read_dreamt_raw_csvs(args.dreamt_data_dir)
    if args.limit_files > 0:
        raw_files = raw_files[: args.limit_files]

    if not raw_files:
        tqdm.write(f"No DREAMT raw CSV files found in {args.dreamt_data_dir}")
        return 0

    first_signal_df = pd.read_csv(raw_files[0], nrows=1)
    modality_tokens = _resolve_tokens_from_arg(args.dreamt_modalities, list(first_signal_df.columns))

    dataset_tag = "dreamt"
    subject_to_records: Dict[str, List[ProcessedRecord]] = defaultdict(list)
    skipped_subjects: set[str] = set()
    processed = 0

    for signal_path in tqdm(raw_files, desc="DREAMT records"):
        record_id = signal_path.stem
        subject_id = infer_subject_id(dataset_tag, record_id)
        if not args.overwrite and _subject_triplet_exists(output_dir, dataset_tag, subject_id):
            if subject_id not in skipped_subjects:
                tqdm.write(f"Outputs exist for DREAMT subject {subject_id}; skipping")
                skipped_subjects.add(subject_id)
            continue

        try:
            rec = process_dreamt_record(
                signal_path=signal_path,
                record_id=record_id,
                modality_tokens=modality_tokens,
                sr=args.target_sr,
                segment_seconds=args.segment_length,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
            )
        except Exception as exc:
            tqdm.write(f"Failed to process DREAMT record {record_id}: {exc}")
            continue

        if rec is None:
            continue

        subject_to_records[rec.subject_id].append(rec)
        processed += 1

    for subject_id, records in sorted(subject_to_records.items()):
        spec_p, ts_p, non_p = _write_subject_triplet(
            output_dir=output_dir,
            dataset_tag=dataset_tag,
            subject_id=subject_id,
            records=records,
        )
        tqdm.write(
            f"Wrote subject {subject_id}: {spec_p.name}, {ts_p.name}, {non_p.name}"
        )

    return processed


# -------------------------------
# CLI
# -------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified DREAMT/Sleep-EDF preprocessing to chunked NPZ files"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["dreamt", "sleep-edf", "both"],
        help="Which dataset pipeline(s) to run",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("preprocessed_npz"),
        help="Directory where chunked NPZ files are written",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute subjects even if subject-level NPZ triplets already exist",
    )
    parser.add_argument(
        "--chunk_size_records",
        type=int,
        default=16,
        help="Deprecated: retained for compatibility; outputs are now written one subject per file",
    )
    parser.add_argument(
        "--limit_files",
        type=int,
        default=0,
        help="Optional cap on records per dataset (0 means no cap)",
    )

    # Shared spectrogram/segmentation params.
    parser.add_argument("--segment_length", type=int, default=30, help="Segment length in seconds")
    parser.add_argument("--target_sr", type=int, default=100, help="Target sampling rate (Hz)")
    parser.add_argument("--n_fft", type=int, default=256, help="STFT n_fft")
    parser.add_argument("--hop_length", type=int, default=64, help="STFT hop_length")
    parser.add_argument("--win_length", type=int, default=256, help="STFT win_length")

    # Sleep-EDF args.
    parser.add_argument(
        "--sleep_edf_psg_dir",
        type=Path,
        default=Path("data/sleepedfx/files/sleep-edfx/1.0.0/sleep-cassette"),
        help="Directory containing Sleep-EDF PSG EDF files",
    )
    parser.add_argument(
        "--sleep_edf_hyp_dir",
        type=Path,
        default=Path("data/sleepedfx/files/sleep-edfx/1.0.0/sleep-cassette"),
        help="Directory containing Sleep-EDF Hypnogram EDF files",
    )
    parser.add_argument(
        "--sleep_edf_modalities",
        type=str,
        default="ALL",
        help="Comma-separated Sleep-EDF channels/canonical names, or ALL",
    )
    parser.add_argument(
        "--max_wake_segments",
        type=int,
        default=60,
        help="Max wake segments retained before/after sleep period",
    )
    parser.add_argument(
        "--sleep_edf20_only",
        action="store_true",
        help="Only process Sleep-EDF-20 SC subset",
    )

    # DREAMT args.
    parser.add_argument(
        "--dreamt_data_dir",
        type=Path,
        default=Path("data/dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.1.0"),
        help="DREAMT root (expects data_100Hz/*.csv) or direct folder containing raw CSV files",
    )
    parser.add_argument(
        "--dreamt_modalities",
        type=str,
        default="ALL",
        help="DREAMT modality tokens/names (e.g. FEC, FLOW,ECG,CHIN, or ALL)",
    )

    args = parser.parse_args()
    if args.chunk_size_records <= 0:
        raise ValueError("--chunk_size_records must be > 0")
    return args


def main() -> None:
    args = parse_args()

    if args.win_length != args.n_fft:
        tqdm.write("Info: overriding --win_length to match --n_fft for consistency")
        args.win_length = args.n_fft

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0

    if args.dataset in {"sleep-edf", "both"}:
        tqdm.write("Starting Sleep-EDF preprocessing...")
        n = run_sleep_edf(args, args.output_dir)
        total_processed += n
        tqdm.write(f"Sleep-EDF complete: processed {n} record(s)")

    if args.dataset in {"dreamt", "both"}:
        tqdm.write("Starting DREAMT preprocessing...")
        n = run_dreamt(args, args.output_dir)
        total_processed += n
        tqdm.write(f"DREAMT complete: processed {n} record(s)")

    tqdm.write(f"Done. Total processed records: {total_processed}")


if __name__ == "__main__":
    main()
