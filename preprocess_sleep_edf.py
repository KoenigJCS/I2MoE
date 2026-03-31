#!/usr/bin/env python3
"""Batch preprocess Sleep-EDF PSG/Hypnogram EDF pairs into spectrogram CSVs."""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from pyedflib import highlevel
from tqdm import tqdm

from train_model import SpectrogramTransformer

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
    "EEG1" : ["EEG_FpzCz"],
    "EEG2": ["EEG_PzOz"],
    "EOG": ["EOG_horizontal"],
    "EMG": ["EMG_submental"],
    "Resp": ["Resp_oro-nasal"],
    "Temp": ["Temp_rectal"],
    "EEG_FpzCz": ["EEG1"],
    "EEG_PzOz": ["EEG2"],
    # "EOG_horizontal": "EOG",
    # "Resp_oro-nasal": "Resp",
    # "EMG_submental": "EMG",
    # "Temp_rectal": "Temp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Sleep-EDF EDF pairs into spectrogram CSVs.")
    parser.add_argument(
        "--edf_dir",
        type=Path,
        default=Path("sleep-edfx/1.0.0/sleep-telemetry"),
        help="Directory containing PSG EDF files (e.g., sleep-cassette)",
    )
    parser.add_argument(
        "--hyp_dir",
        type=Path,
        default=Path("sleep-edfx/1.0.0/sleep-telemetry"),
        help="Directory containing Hypnogram EDF files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("sleep-edf/preprocessed"),
        help="Output directory for spectrogram CSVs",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="ALL",
        help="Comma-separated channel labels or canonical names to extract, or ALL to include every channel",
    )
    parser.add_argument("--segment_length", type=int, default=30, help="Segment length in seconds")
    parser.add_argument("--target_sr", type=int, default=100, help="Target sampling rate (Hz)")
    parser.add_argument("--n_fft", type=int, default=256, help="n_fft for STFT")
    parser.add_argument("--hop_length", type=int, default=64, help="hop_length for STFT")
    parser.add_argument("--win_length", type=int, default=256, help="win_length for STFT")
    parser.add_argument(
        "--max_wake_segments",
        type=int,
        default=60,
        help="Max number of wake segments kept before/after sleep onset (30s segments)",
    )
    parser.add_argument("--limit_files", type=int, default=0, help="Process at most this many EDF pairs (0 = no limit)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output CSVs exist")
    parser.add_argument(
        "--sleep_edf20_only",
        action="store_true",
        help="Only process the Sleep-EDF-20 SC subset (subjects 00-19, nights 1/2; missing SC4132 is naturally skipped)",
    )
    return parser.parse_args()


SLEEP_EDF20_SC_PSG_RE = re.compile(r"^SC4(0[0-9]|1[0-9])[12]E0-PSG\.edf$")


def is_sleep_edf20_sc_psg(path: Path) -> bool:
    return bool(SLEEP_EDF20_SC_PSG_RE.match(path.name))


def find_hypnogram_for_psg(psg_path: Path, hyp_dir: Path) -> Path | None:
    record_id = psg_path.stem.replace("-PSG", "")
    # First, try exact (some datasets may use the same stem)
    exact = hyp_dir / f"{record_id}-Hypnogram.edf"
    if exact.exists():
        return exact
    # Sleep-EDFx convention: trailing '0' in PSG stem is replaced by scorer letter in hypnogram
    if record_id.endswith("0"):
        stem_prefix = record_id[:-1]
        candidates = sorted(hyp_dir.glob(f"{stem_prefix}?-Hypnogram.edf"))
        if candidates:
            return candidates[0]
    # Last-chance broad search by common prefix
    candidates = sorted(hyp_dir.glob(f"{record_id[:7]}*-Hypnogram.edf"))
    if candidates:
        return candidates[0]
    return None


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
    print(f"Warning: Unrecognized sleep stage label '{label}'; mapping to 'Missing'")
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


def read_hypnogram_annotations(hyp_path: Path) -> List[Tuple[float, float, str]]:
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


def build_stage_series(annotations: List[Tuple[float, float, str]], total_samples: int, target_sr: int) -> np.ndarray:
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
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[str]]:
    signals, signal_headers, header = highlevel.read_edf(str(edf_path), digital=False)
    # print(f"Loaded {edf_path.name} with {len(signals)} signals")
    # print(f"Signal headers: {[hdr['label'] for hdr in signal_headers]}")
    if include_all:
        raw_labels = [hdr["label"].strip() for hdr in signal_headers]
        ordered_modalities = make_unique_labels(raw_labels)
        matchers = {}
    else:
        ordered_modalities, matchers = build_requested_modalities(requested_modalities)
    data: Dict[str, np.ndarray] = {}
    srs: Dict[str, float] = {}
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
        sr = float(sr)
        signal = signals[idx]
        data[matched] = resample_signal(signal, sr, target_sr)
        srs[matched] = sr
    return data, srs, ordered_modalities


def compute_spectrogram_segments(signal: np.ndarray, processor: SpectrogramTransformer, segment_samples: int) -> np.ndarray:
    usable = (len(signal) // segment_samples) * segment_samples
    if usable == 0:
        return np.empty((0, 0), dtype=np.float32)
    tensor = torch.tensor(signal[:usable], dtype=torch.float32)
    segments = tensor.view(-1, segment_samples)
    specs = [processor.preprocess(seg) for seg in segments]
    flattened = [spec.numpy().flatten() for spec in specs]
    if not flattened:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(flattened, axis=0)


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
        post_idx = np.where(labels[last_sleep + 1:] == "W")[0] + last_sleep + 1
        if post_idx.size > max_wake_segments:
            drop = post_idx[max_wake_segments:]
            trimmed[drop] = False
    return trimmed


def process_record(edf_path: Path, hyp_path: Path, args: argparse.Namespace, processor: SpectrogramTransformer) -> None:
    raw_modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    include_all = any(m.upper() in {"ALL", "*"} for m in raw_modalities) or not raw_modalities
    if include_all:
        raw_modalities = []
    modality_data, original_srs, ordered_modalities = load_modalities(
        edf_path, raw_modalities, args.target_sr, include_all
    )
    if not modality_data:
        tqdm.write(f"No requested modalities found in {edf_path.name}; skipping")
        return
    min_len = min(len(arr) for arr in modality_data.values())
    segment_samples = args.target_sr * args.segment_length
    usable_samples = (min_len // segment_samples) * segment_samples
    if usable_samples == 0:
        tqdm.write(f"Record {edf_path.name} shorter than one {args.segment_length}s window; skipping")
        return
    for mod in modality_data:
        modality_data[mod] = modality_data[mod][:usable_samples]
    annotations = read_hypnogram_annotations(hyp_path)
    stage_series = build_stage_series(annotations, usable_samples, args.target_sr)
    timestamps = np.arange(usable_samples) / args.target_sr
    num_segments = usable_samples // segment_samples
    stage_series = stage_series[: usable_samples]
    stage_segments = stage_series.reshape(num_segments, segment_samples)
    segment_labels = []
    for seg in stage_segments:
        labels, counts = np.unique(seg, return_counts=True)
        segment_labels.append(labels[int(np.argmax(counts))])
    keep_mask = trim_wake_segments(segment_labels, args.max_wake_segments)
    if not np.any(keep_mask):
        tqdm.write(f"All segments filtered (non-base classes) for {edf_path.name}; skipping")
        tqdm.write(f"Segment label distribution: {pd.Series(segment_labels).value_counts().to_dict()}")
        return
    keep_timepoints = np.repeat(keep_mask, segment_samples)[:usable_samples]
    stage_series = stage_series[keep_timepoints]
    timestamps = timestamps[keep_timepoints]

    processed_columns: Dict[str, np.ndarray] = {}
    for mod in ordered_modalities:
        signal = modality_data.get(mod)
        if signal is None:
            continue
        segments = compute_spectrogram_segments(signal, processor, segment_samples)
        if segments.size == 0:
            tqdm.write(f"Modality {mod} in {edf_path.name} produced no usable spectrogram data; skipping record")
            return
        if segments.shape[0] != num_segments:
            num_segments = min(num_segments, segments.shape[0])
            keep_mask = keep_mask[:num_segments]
        segments = segments[:num_segments]
        segments = segments[keep_mask]
        if segments.size == 0:
            tqdm.write(f"All segments filtered for {edf_path.name} after modality {mod}; skipping record")
            return
        processed_columns[mod] = segments.reshape(-1)

    processed_df = pd.DataFrame(processed_columns)
    non_signal_df = pd.DataFrame({
        "TIMESTAMP": timestamps,
        "Sleep_Stage": stage_series,
    })

    base = edf_path.stem.replace("-PSG", "")
    output_main = args.output_dir / f"preprocessed_{base}.csv"
    output_meta = args.output_dir / f"preprocessed_non_signal_{base}.csv"
    processed_df.to_csv(output_main, index=False)
    non_signal_df.to_csv(output_meta, index=False)
    tqdm.write(f"Saved {output_main.name} and {output_meta.name} from {edf_path.name} and {hyp_path.name}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = SpectrogramTransformer(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        sr=args.target_sr,
        db_min=-80.0,
    )

    psg_files = sorted(args.edf_dir.glob("*-PSG.edf"))
    if args.sleep_edf20_only:
        psg_files = [p for p in psg_files if is_sleep_edf20_sc_psg(p)]
        print(f"Sleep-EDF-20 filter enabled: {len(psg_files)} PSG file(s) matched in {args.edf_dir}")
    if not psg_files:
        print(f"No PSG EDF files found in {args.edf_dir}")
        # try .hyp and .red

        sys.exit(1)

    # pair -PSG and -Hypnograms 
    processed = 0
    for psg_path in tqdm(psg_files, desc="Records"):
        record_id = psg_path.stem.replace("-PSG", "")
        hyp_path = find_hypnogram_for_psg(psg_path, args.hyp_dir)
        hyp_name = hyp_path.name if hyp_path is not None else f"{record_id}-Hypnogram.edf"
        output_main = args.output_dir / f"preprocessed_{record_id}.csv"
        output_meta = args.output_dir / f"preprocessed_non_signal_{record_id}.csv"
        if not args.overwrite and output_main.exists() and output_meta.exists():
            tqdm.write(f"Outputs exist for {record_id}; skipping (use --overwrite to recompute)")
            processed += 1
            if args.limit_files and processed >= args.limit_files:
                break
            continue
        if hyp_path is None or not hyp_path.exists():
            tqdm.write(f"Missing Hypnogram {hyp_name}; skipping record")
            continue
        try:
            process_record(psg_path, hyp_path, args, processor)
            processed += 1
        except Exception as exc:
            tqdm.write(f"Failed to process {record_id}: {exc}")
        if args.limit_files and processed >= args.limit_files:
            break
    print(f"Finished processing {processed} record(s).")


if __name__ == "__main__":
    main()
