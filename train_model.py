import os
import sys
import json
import hashlib
from datetime import datetime
import numpy as np
from typing import Optional, List, Dict
import argparse
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.utils.clip_grad import clip_grad_norm_
from ordered_set import OrderedSet
from denoising_diffusion_1d import normalize_signal_1d, SpectrogramNormalizer

from models.SleepViTModels import InterEpochTransformer

predict_only_wake_sleep = False

TB_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    TB_AVAILABLE = True
except Exception:
    try:
        from torch.utils.tensorboard.writer import SummaryWriter  # type: ignore
        TB_AVAILABLE = True
    except Exception:
        TB_AVAILABLE = False

from models.sleep_stage_model import SleepStageClassifier
from models.temporal_head import TemporalHead
from torchvision.models.feature_extraction import create_feature_extractor

# import librosa

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

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_seq', type=int, default=10)
    parser.add_argument('--data_path', type=str, default="dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.1.0/data_100Hz/")
    parser.add_argument('--preprocess_path', type=str, default="presave/")
    parser.add_argument('--save_path', type=str, default="result/")
    parser.add_argument('--save_preprocessed', action='store_true', help="Whether to save the preprocessed data")
    parser.add_argument('--load_preprocessed', action='store_true', help="Whether to load the preprocessed data if already saved")
    parser.add_argument('--max_preprocessed_files', type=int, default=0, help='Limit number of cached preprocessed CSVs to load (0 = load all)')
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--dry_run', action='store_true', help='Run a quick forward/backward pass on a tiny subset to verify shapes')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of sleep stages')
    # STFT and modalities
    parser.add_argument('--modalities', type=str, default='FLOW,ECG,CHIN', help='Comma-separated modality column names to use, e.g., IBI,SpO2,ACC_X')
    parser.add_argument('--n_fft', type=int, default=256, help='STFT n_fft for spectrograms')
    parser.add_argument('--hop_length', type=int, default=64 , help='STFT hop length')
    parser.add_argument('--win_length', type=int, default=256 , help='STFT window length')
    parser.add_argument('--sr', type=int, default=100, help='Sampling rate for the input signals')
    parser.add_argument('--limit_batches', type=int, default=0, help='Limit number of batches per epoch for quick experiments (0 = no limit)')
    parser.add_argument('--use_weighted_sampler', action='store_true', help='Use WeightedRandomSampler to mitigate class imbalance in training')
    parser.add_argument('--track_energy', action='store_true', help='Save per-split energy/uncertainty arrays to disk for analysis')
    parser.add_argument('--energy_threshold', type=float, default=None, help='Drop predictions with energy above this value when reporting accuracy (higher energy = more uncertain)')
    parser.add_argument('--energy_step', type=float, default=-0.5, help='Step size for energy threshold curve (negative values step downward)')
    parser.add_argument('--energy_levels', type=str, default='', help='Comma-separated coverage percents for reporting energy/accuracy (e.g., 95,80,75)')
    parser.add_argument('--visualize_model', action='store_true', help='Generate t-SNE feature visualization and save to disk')
    parser.add_argument('--tsne_max_samples', type=int, default=2000, help='Max samples for t-SNE visualization')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0, help='t-SNE perplexity')
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard log directory (if TB installed)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Clip gradient norm to this value (0 disables)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['none', 'cosine', 'step'], help='Learning-rate annealing strategy')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate for cosine annealing')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Step size (epochs) for step LR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Multiplicative decay factor for step LR scheduler')
    parser.add_argument(
        '--model',
        type=str,
        default='convnext',
        choices=['convnext', 'simple_cnn', 'transformer', 'resnet', 'resnet152', 'sleepvit'],
        help='Model architecture to use (convnext, simple_cnn, transformer, resnet/resnet152, sleepvit)'
    )
    parser.add_argument('--stage', type=str, default='full5', choices=['full5', 'binary', 'sleep4', 'n1', 'sleep3', 'fallback5'], help='Training stage: full5 (W,N1,N2,N3,REM), binary (W/S), n1 (N1 vs non-N1 sleep), sleep3 (N2/N3/REM), sleep4 (legacy N1/N2/N3/REM), or fallback5 (same as full5)')
    parser.add_argument('--convnext_size', type=str, default='large', choices=['tiny', 'small', 'large'], help='ConvNeXt size for single-stage training')
    parser.add_argument('--binary_convnext_size', type=str, default='large', choices=['tiny', 'small', 'large'], help='ConvNeXt size for binary wake/sleep stage')
    parser.add_argument('--sleep_convnext_size', type=str, default='large', choices=['tiny', 'small', 'large'], help='Legacy sleep-stage size (used for sleep4 and as fallback for sleep3 if unspecified)')
    parser.add_argument('--n1_convnext_size', type=str, default='small', choices=['tiny', 'small', 'large'], help='ConvNeXt size for N1 discriminator stage')
    parser.add_argument('--sleep3_convnext_size', type=str, default='large', choices=['tiny', 'small', 'large'], help='ConvNeXt size for sleep3 stage (N2/N3/REM)')
    parser.add_argument('--fallback_convnext_size', type=str, default='large', choices=['tiny', 'small', 'large'], help='ConvNeXt size for fallback 5-class stage')
    parser.add_argument('--cascade_eval', action='store_true', help='Run cascade inference using binary + n1 + sleep3 + fallback models')
    parser.add_argument('--cascade_train', action='store_true', help='Train/calibrate cascade routing after binary/n1/sleep3/fallback checkpoints are pretrained')
    parser.add_argument('--binary_checkpoint', type=str, default='result/binary/best.pt', help='Checkpoint for binary wake/sleep model (cascade eval)')
    parser.add_argument('--sleep_checkpoint', type=str, default='result/sleep4/best.pt', help='Legacy sleep4 checkpoint (backward compatibility alias)')
    parser.add_argument('--n1_checkpoint', type=str, default='result/n1/best.pt', help='Checkpoint for N1 discriminator model (cascade eval)')
    parser.add_argument('--sleep3_checkpoint', type=str, default='result/sleep3/best.pt', help='Checkpoint for sleep3 model (N2/N3/REM) (cascade eval)')
    parser.add_argument('--fallback_checkpoint', type=str, default='result/fallback/best.pt', help='Checkpoint for fallback 5-class model (cascade eval)')
    parser.add_argument('--cascade_checkpoint', type=str, default='', help='Path to cascade checkpoint containing calibrated routing threshold')
    parser.add_argument('--cascade_metric', type=str, default='kappa', choices=['kappa', 'f1', 'acc'], help='Primary metric for cascade threshold calibration')
    parser.add_argument('--cascade_threshold_points', type=int, default=61, help='Number of quantile threshold candidates for cascade calibration')
    parser.add_argument('--cascade_max_fallback_rate', type=float, default=0.20, help='Target maximum fraction routed to fallback during cascade calibration')
    parser.add_argument('--binary_energy_threshold_start', type=float, default=-3.0, help='Initial energy threshold for binary confidence (warmup)')
    parser.add_argument('--binary_energy_warmup_epochs', type=int, default=5, help='Epochs to keep the initial binary energy threshold')
    parser.add_argument('--binary_energy_target_acc', type=float, default=0.60, help='Target accuracy for energy-based confidence threshold')
    parser.add_argument('--save_routing_weights', action='store_true', help='In binary stage, save per-sample routing likelihoods for n1/sleep3/fallback models')
    parser.add_argument('--use_routing_weights', action='store_true', help='In n1/sleep3/fallback stages, upweight samples likely routed there by binary model')
    parser.add_argument('--routing_weights_path', type=str, default='', help='Path to routing-weights JSON (defaults to <save_path>/routing_weights.json when saving, result/binary/routing_weights.json when using)')
    parser.add_argument('--routing_weight_power', type=float, default=1.0, help='Exponent applied to routing weights before sampling')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of dataset used for validation (0 disables)')
    parser.add_argument('--test_split', type=float, default=0.1, help='Fraction of dataset reserved for test evaluation (0 disables)')
    parser.add_argument('--split_mode', type=str, default='sample', choices=['sample', 'subject'], help='How to form dataset splits (per-sample or per-subject)')
    parser.add_argument('--split_manifest_path', type=str, default='', help='Path to split manifest JSON. If it exists, splits are loaded from it; otherwise generated splits can be written to it.')
    parser.add_argument('--write_split_manifest', action='store_true', help='Write generated train/val/test split manifest to --split_manifest_path')
    parser.add_argument('--strict_split_manifest', action='store_true', help='Fail if any manifest sample key is missing in current dataset')
    parser.add_argument('--checkpoint_path', type=str, default='result/best.pt', help='Checkpoint path to load for evaluation/testing')
    parser.add_argument('--test_only', action='store_true', help='Load --checkpoint_path and evaluate on test/val split without training')
    parser.add_argument('--temporal', type=int, default=0, help='Temporal context radius N (uses 2N+1 samples); 0 disables')
    parser.add_argument('--temporal_only', action='store_true', help='Train temporal head only using a frozen base model')
    parser.add_argument('--base_checkpoint', type=str, default='result/best_base.pt', help='Base model checkpoint for temporal-only training')
    parser.add_argument('--temporal_checkpoint', type=str, default='result/best_temporal.pt', help='Temporal head checkpoint to resume in temporal-only training')
    parser.add_argument('--temporal_hidden', type=int, default=512, help='Hidden size for temporal head MLP')
    parser.add_argument('--temporal_dropout', type=float, default=0.2, help='Dropout for temporal head MLP')
    # Domain adaptation (MMD / DAM-Net)
    parser.add_argument('--domain_adapt', type=str, default='none', choices=['none', 'mmd', 'damnet'], help='Domain adaptation strategy')
    parser.add_argument('--target_preprocess_path', type=str, default='shhs/preprocessed', help='Target preprocessed spectrogram directory')
    parser.add_argument('--target_data_path', type=str, default='shhs', help='Target raw data path (unused when preprocessed exists)')
    parser.add_argument('--target_max_preprocessed_files', type=int, default=0, help='Limit number of target cached preprocessed CSVs to load (0 = load all)')
    parser.add_argument('--target_batch_size', type=int, default=0, help='Batch size for target domain (0 = use --batch_size)')
    parser.add_argument('--mmd_weight', type=float, default=0.1, help='Weight for MMD adaptation loss')
    parser.add_argument('--mmd_sigmas', type=str, default='1,2,4,8,16', help='Comma-separated Gaussian kernel sigmas for MMD')
    parser.add_argument('--mmd_classwise', action='store_true', default=True, help='Use discriminative class-wise MMD (with pseudo-labels)')
    parser.add_argument('--no_mmd_classwise', action='store_true', help='Disable class-wise MMD (use global MMD only)')
    parser.add_argument('--micro_label_ratio', type=float, default=0.01, help='Fraction of target samples used as micro-labeled set (DAM-Net)')
    parser.add_argument('--micro_label_count', type=int, default=0, help='Number of target samples used as micro-labeled set (overrides ratio)')
    parser.add_argument('--micro_lr', type=float, default=0.0, help='Learning rate for micro-labeled fine-tuning (0 = lr*0.1)')
    parser.add_argument('--loss_mode', type=str, default='supcon', choices=['ce', 'supcon'], help='Loss mode: cross-entropy or supervised contrastive')
    parser.add_argument('--supcon_temp', type=float, default=7e-3, help='Temperature for supervised contrastive loss')
    parser.add_argument('--supcon_weight', type=float, default=1.0, help='Weight for supervised contrastive loss')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='Weight for cross-entropy loss (used with supcon)')

    args = parser.parse_args()
    if args.no_mmd_classwise:
        args.mmd_classwise = False

    stage_mode = args.stage
    label_mode = 'full5' if stage_mode in ['full5', 'fallback5'] else stage_mode
    if args.cascade_eval or args.cascade_train:
        stage_mode = 'full5'
        label_mode = 'full5'
    predict_only_wake_sleep = label_mode == 'binary'
    if stage_mode == 'binary':
        args.convnext_size = args.binary_convnext_size
    elif stage_mode == 'n1':
        args.convnext_size = args.n1_convnext_size
    elif stage_mode == 'sleep3':
        args.convnext_size = args.sleep3_convnext_size
    elif stage_mode == 'sleep4':
        args.convnext_size = args.sleep_convnext_size
    elif stage_mode == 'fallback5':
        args.convnext_size = args.fallback_convnext_size

    def _parse_energy_levels(levels: str):
        if not levels:
            return []
        cleaned = levels.replace(',', ' ')
        vals = []
        for chunk in cleaned.split():
            try:
                val = float(chunk)
            except Exception:
                continue
            if 0.0 < val <= 100.0:
                vals.append(val)
        return sorted(set(vals), reverse=True)

    energy_levels = _parse_energy_levels(args.energy_levels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Set seed
    torch.manual_seed(1)
    np.random.seed(1)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.save_preprocessed and not os.path.exists(args.preprocess_path):
        os.makedirs(args.preprocess_path)

    def _run_log_path() -> str:
        return os.path.join(args.save_path, 'run_data.log')

    def _append_run_log(payload: dict):
        try:
            os.makedirs(args.save_path, exist_ok=True)
            with open(_run_log_path(), 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload) + '\n')
        except Exception as e:
            print('Warning: failed to write run_data.log:', e)

    run_id = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    _append_run_log({
        'event': 'run_start',
        'run_id': run_id,
        'timestamp_utc': run_id,
        'args': vars(args),
        'cwd': os.getcwd()
    })

    if args.save_preprocessed:
        #load data for preprocessing and cache to CSVs
        print("========= Preprocess & Save =========")
        files = [x for x in os.listdir(args.data_path) if x.endswith('.csv')]
        pbar = tqdm(total=len(files), desc="Files", position=0)

        processed_data_list = []
        files.sort()

        for file in files:
            # Completion bar
            if args.load_preprocessed and os.path.exists(os.path.join(args.preprocess_path, f"preprocessed_{file}")):
                tqdm.write(f"Preprocessed file exists, skipping: preprocessed_{file}")
                pbar.update(1)
                continue
            tqdm.write(f"Processing file: {file}")
            person_file = pd.read_csv(os.path.join(args.data_path, file), header=0)
            # always remove first second of data
            # find row with 1.0 in TIMESTAMP
            offset = person_file[person_file['TIMESTAMP'] == 1.0].index[0]
            person_file = person_file.iloc[offset:, :]  # remove rows until labels are aligned with segments*frequency
            processed_file = pd.DataFrame({})
            processed_file_non_signal = pd.DataFrame({})
            colbar = tqdm(total=len(person_file.columns)*2, desc="Cols", position=1, leave=False)
            # remove rows with bad labels "P" and "Missing"
            series = person_file['Sleep_Stage'].to_numpy()
                        # predict_only_wake_sleep = True


            invalid_indices = OrderedSet([i for i, v in enumerate(series) if v in ['P', 'Missing']])
            pruned_file = pd.DataFrame()
            removed_ranges = []
            i=0
            while i < len(series):
                if(i not in invalid_indices and all(x not in invalid_indices for x in range(i, i+args.sr*30))):
                    pruned_file = pd.concat([pruned_file, person_file.iloc[i:i+args.sr*30]], axis=0)
                    i+=args.sr*30
                else:
                    i+=1
            person_file = pruned_file
            for column in person_file.columns:
                processed_data = pd.DataFrame({})
                if column not in ['TIMESTAMP', 'Sleep_Stage', 'Mixed_Apnea', 'Obstructive_Apnea', 'Central_Apnea', 'Hypopnea']:
                    series = person_file[column].to_numpy(dtype=np.float32)
                    data = torch.tensor(series, dtype=torch.float32)
                    data_processor = SpectrogramTransformer(sr=args.sr*4) 
                    data_processor.n_fft = args.n_fft
                    data_processor.hop_length = args.hop_length
                    data_processor.win_length = args.win_length
                    # trim to full segments of 30*64 to avoid short remainders
                    win = 30*args.sr
                    usable = (len(data) // win) * win
                    data = data[:usable]
                    # split data into segments of 30*64
                    data = data.split(win)
                    # preprocess each full segment
                    tmp_processed_data = [data_processor.preprocess(segment) for segment in data]
                    colbar.update(1)
                    processed_data = pd.DataFrame(np.concatenate([segment.numpy().flatten() for segment in tmp_processed_data]))
                    colbar.update(1)
                    processed_file[column] = processed_data                    
                else:
                    colbar.update(2)
                    # only save one label per window size TODO
                    series = person_file[column].to_numpy()
                    if(column == 'Sleep_Stage'):
                        labels = []
                        tries = 0
                        i=0
                        for _ in range(0, len(series), args.sr):
                            window = series[i:i+args.sr]
                            if len(window) == 0:
                                i+=args.sr
                                continue
                            # majority label in the window
                            counts = {}
                            for label in window:
                                if label == 'P' or label == 'Missing':
                                    raise ValueError(f"Invalid label '{label}' found in window: {window}")
                                if label in counts:
                                    counts[label] += 1
                                elif len(counts) > 0:
                                    # already have a different label, choose the one with higher count
                                    max_label = max(counts, key=counts.get)
                                    if counts[max_label] >= 2:
                                        labels.append(max_label)
                                    else:
                                        # no majority, error
                                        raise ValueError(f"No majority label in window: {window}")
                                    counts = {label: 1}
                                else:
                                    counts[label] = 1
                            i+=args.sr
                    processed_file_non_signal[column] = person_file[column]

            pbar.update(1)
            if args.save_preprocessed:
                processed_file.to_csv(os.path.join(args.preprocess_path, f"preprocessed_{file}"), index=False)
                processed_file_non_signal.to_csv(os.path.join(args.preprocess_path, f"preprocessed_non_signal_{file}"), index=False)
                tqdm.write(f"Saved preprocessed file: preprocessed_{file}")
            colbar.close()

        pbar.close()
    else:
        print("Skipping preprocessing step (use --save_preprocessed to generate cached spectrograms).")

    # Load the model
    print("========= Build Dataset & Model =========")

    class SleepDataset(Dataset):
        def __init__(self, preprocess_dir: str, fallback_raw_dir: str, frequency: int = 64, segment_len_s: int = 30, num_modalities: int = 1, max_preprocessed_files: Optional[int] = None):
            super().__init__()
            self.frequency = frequency
            self.segment_len = segment_len_s
            self.num_modalities = num_modalities
            self.modalities = [m.strip() for m in args.modalities.split(',') if m.strip()]
            self.n_fft = args.n_fft
            self.hop_length = args.hop_length
            self.win_length = args.win_length
            self.files = []
            self.labels = []
            self.sample_subjects = []
            self.sample_keys = []
            # Global target shapes for (F, T) across the entire dataset
            pad = self.n_fft // 2  # center=True in STFT
            L = self.frequency * self.segment_len
            self.F_target = self.n_fft // 2 + 1
            self.T_target = 1 + max(0, (L + 2 * pad - self.n_fft) // max(1, self.hop_length))

            def pad_or_crop_2d(spec: np.ndarray, target_f: int, target_t: int) -> np.ndarray:
                # spec shape (F, T). Crop or pad with zeros to (target_f, target_t)
                F, T = spec.shape
                # Crop
                spec_c = spec[:min(F, target_f), :min(T, target_t)]
                # Pad if needed
                out = np.zeros((target_f, target_t), dtype=spec_c.dtype)
                out[:spec_c.shape[0], :spec_c.shape[1]] = spec_c
                return out

            pre_csvs = []
            if os.path.isdir(preprocess_dir):
                pre_csvs = [os.path.join(preprocess_dir, f) for f in os.listdir(preprocess_dir) if f.startswith('preprocessed_') and f.endswith('.csv') and not f.startswith('preprocessed_non_signal_')]
            pre_csvs.sort()
            max_files = args.max_preprocessed_files if max_preprocessed_files is None else max_preprocessed_files
            if max_files and max_files > 0:
                pre_csvs = pre_csvs[:max_files]
            if len(pre_csvs) == 0:
                raise MemoryError("No preprocessed files found.")
            else:
                # Use preprocessed spectrogram csvs that were saved as flattened streams per modality column
                tqdm.write(f"Loading preprocessed data from {len(pre_csvs)} files in {preprocess_dir}")
                loadbar = tqdm(total=len(pre_csvs), desc="Loading preprocessed data", position=0)
                self.samples = []

                def infer_subject_id(filename: str) -> str:
                    base_name = os.path.splitext(filename)[0]
                    return base_name.split('_')[0] if '_' in base_name else base_name

                for pcsv in pre_csvs:
                    pdf = pd.read_csv(pcsv)
                    base = os.path.basename(pcsv).removeprefix('preprocessed_')
                    self.files.append(base)
                    subject_id = infer_subject_id(base)
                    non_sig_path = os.path.join(preprocess_dir, f"preprocessed_non_signal_{base}")
                    labels_arr = None
                    # Pull labels and compute per-segment labels
                    if os.path.exists(non_sig_path):
                        ndf = pd.read_csv(non_sig_path)
                        if 'Sleep_Stage' in ndf.columns:
                            labels_full = ndf['Sleep_Stage'].to_numpy()

                            def canonicalize_label(raw_label) -> Optional[str]:
                                label = str(raw_label).strip().upper()
                                if label in ['R', 'REM']:
                                    return 'REM'
                                if label in ['W', 'N1', 'N2', 'N3']:
                                    return label
                                if label in ['P', 'MISSING']:
                                    return None
                                return label

                            def map_labels_full(raw_labels: np.ndarray) -> np.ndarray:
                                if raw_labels.dtype.kind in {'U', 'S', 'O'}:
                                    mapped = []
                                    for lbl in raw_labels:
                                        canon = canonicalize_label(lbl)
                                        if label_mode == 'binary':
                                            mapped.append('W' if canon in ['W', None] else 'S')
                                        else:
                                            mapped.append(canon if canon is not None else 'Missing')
                                    raw_labels = np.asarray(mapped, dtype=object)
                                    if label_mode == 'binary':
                                        label_map = {'W': 0, 'S': 1}
                                    elif label_mode == 'n1':
                                        label_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
                                    elif label_mode == 'sleep3':
                                        label_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
                                    else:
                                        label_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
                                    return np.asarray([label_map.get(v, 0) for v in raw_labels], dtype=np.int64)
                                # assume numeric labels already in 0..4 with 0=Wake
                                labels_int = np.asarray(raw_labels, dtype=np.int64)
                                if label_mode == 'binary':
                                    return (labels_int != 0).astype(np.int64)
                                return labels_int

                            labels_full = map_labels_full(labels_full)
                            win = self.frequency * self.segment_len
                            usable = (len(labels_full) // win) * win
                            if usable == 0:
                                loadbar.update(1)
                                continue
                            labels_full = labels_full[:usable]
                            seg_labels = labels_full.reshape(-1, win)
                            labels_arr_full = np.array([np.bincount(seg.astype(int)).argmax() for seg in seg_labels], dtype=np.int64)
                            labels_arr = labels_arr_full.copy()
                    # Build per-modality 3D arrays (N, F, T) from flattened streams for requested modalities only
                    # Ensure all requested modalities exist in this file; otherwise skip for consistency across dataset
                    ordered_mods = [m for m in self.modalities if m in pdf.columns]
                    if len(ordered_mods) != len(self.modalities):
                        tqdm.write(f"Skipping {pcsv} because not all requested modalities are present: wanted {self.modalities}, found {ordered_mods}")
                        loadbar.update(1)
                        continue
                    pdf = pdf.loc[:, ordered_mods]
                    mod_streams = {}
                    for col in ordered_mods:
                        col_vals = pdf[col].to_numpy(dtype=np.float32).reshape(-1)
                        col_vals = np.nan_to_num(col_vals, nan=0.0, posinf=0.0, neginf=0.0)
                        mod_streams[col] = col_vals

                    # Determine number of segments N
                    if labels_arr is not None:
                        N = int(labels_arr.shape[0])
                    else:
                        # Infer N by aligning on greatest common divisor across modality lengths and expected per-segment samples
                        # We assume each segment chunk_len is constant per modality
                        L = self.frequency * self.segment_len  # 64*30 by default
                        # Expected STFT frames with center=True pad=n_fft//2
                        pad = args.n_fft // 2
                        t_frames = 1 + max(0, (L + 2 * pad - args.n_fft) // max(1, args.hop_length))
                        f_bins_candidates = [args.n_fft // 2 + 1, args.n_fft]
                        # Guess chunk_len as one of these
                        guess_chunk = None
                        for fb in f_bins_candidates:
                            guess = fb * t_frames
                            if guess > 0:
                                guess_chunk = guess
                                break
                        # Use the shortest stream and round down
                        shortest = min(len(v) for v in mod_streams.values())
                        N = max(1, shortest // max(1, (guess_chunk or shortest)))

                    # Build (N, M, F, T) with global targets (self.F_target, self.T_target)
                    per_mod_segments = []
                    usable = self.F_target * self.T_target
                    totals = {col: len(stream) for col, stream in mod_streams.items()}
                    chunk_lens = {col: (totals[col] // N) for col in ordered_mods}
                    min_chunk = min(chunk_lens.values())
                    if min_chunk <= 0:
                        loadbar.update(1)
                        continue
                    for col in ordered_mods:
                        stream = mod_streams[col]
                        # allocate (N, F, T) and fill per segment with truncation or zero-pad as needed
                        segs = np.zeros((N, self.F_target, self.T_target), dtype=np.float32)
                        c_len = chunk_lens[col]
                        for i in range(N):
                            part = stream[i*c_len:(i+1)*c_len]
                            part = np.nan_to_num(part, nan=0.0, posinf=0.0, neginf=0.0)
                            fill = min(len(part), usable)
                            if fill > 0:
                                segs[i].reshape(-1)[:fill] = part[:fill]
                            # safe per-segment normalize
                            mn = float(np.min(segs[i]))
                            mx = float(np.max(segs[i]))
                            if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                                segs[i] = (segs[i] - mn) / (mx - mn + 1e-6)
                            else:
                                segs[i] = 0.0
                        per_mod_segments.append(segs)
                    fts = (self.F_target, self.T_target)

                    if len(per_mod_segments) == 0 or fts is None:
                        loadbar.update(1)
                        continue

                    # Stack modalities: (N, M, F, T)
                    X = np.stack(per_mod_segments, axis=1)
                    segment_ids = np.arange(X.shape[0], dtype=np.int64)
                    # Labels
                    if labels_arr is None:
                        y = np.zeros(N, dtype=np.int64)
                    else:
                        # ensure same N
                        N2 = min(N, labels_arr.shape[0])
                        X = X[:N2]
                        y = labels_arr[:N2]
                        segment_ids = segment_ids[:N2]
                    if label_mode in ['sleep4', 'n1', 'sleep3'] and labels_arr is not None:
                        # Remove wake segments for sleep-only stages.
                        sleep_mask = y != 0
                        X = X[sleep_mask]
                        y = y[sleep_mask]
                        segment_ids = segment_ids[sleep_mask]
                        if label_mode == 'sleep4':
                            y = y - 1  # N1/N2/N3/REM -> 0..3
                        elif label_mode == 'n1':
                            y = (y == 1).astype(np.int64)  # N1 vs non-N1 sleep
                        elif label_mode == 'sleep3':
                            non_n1_mask = y != 1
                            X = X[non_n1_mask]
                            y = y[non_n1_mask]
                            segment_ids = segment_ids[non_n1_mask]
                            remap = {2: 0, 3: 1, 4: 2}  # N2/N3/REM
                            y = np.asarray([remap.get(int(lbl), 0) for lbl in y], dtype=np.int64)

                    # Register samples
                    for i in range(X.shape[0]):
                        self.samples.append((X[i], int(y[i])))
                        self.sample_subjects.append(subject_id)
                        self.sample_keys.append(f"{base}::seg{int(segment_ids[i])}")
                    loadbar.update(1)

                loadbar.close()
                self.from_preprocessed = True

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            x, y = self.samples[idx]
            # Ensure shape (M, F, T)
            if x.ndim == 2:
                x = x[None, ...]
            # Create a fresh tensor with its own resizable storage to avoid DataLoader shared-memory resize errors
            x_t = torch.tensor(x.copy(), dtype=torch.float32)
            y_t = torch.tensor(int(y), dtype=torch.long)
            return x_t, y_t

        def get_sample(self, idx: int):
            x, y = self.samples[idx]
            if x.ndim == 2:
                x = x[None, ...]
            return x, int(y)

    class TemporalContextDataset(Dataset):
        def __init__(self, base_dataset: SleepDataset, indices=None, radius: int = 0):
            super().__init__()
            self.base = base_dataset
            self.radius = max(0, int(radius))
            self.indices = list(indices) if indices is not None else None
            self.allowed_set = set(self.indices) if self.indices is not None else None

        def __len__(self):
            return len(self.indices) if self.indices is not None else len(self.base)

        def _context_indices(self, base_idx: int):
            if self.radius <= 0:
                return [base_idx]
            indices = []
            subject = self.base.sample_subjects[base_idx] if hasattr(self.base, 'sample_subjects') else None
            for offset in range(-self.radius, self.radius + 1):
                cand = base_idx + offset
                if cand < 0 or cand >= len(self.base):
                    indices.append(base_idx)
                    continue
                if self.allowed_set is not None and cand not in self.allowed_set:
                    indices.append(base_idx)
                    continue
                if subject is not None:
                    if self.base.sample_subjects[cand] != subject:
                        indices.append(base_idx)
                        continue
                indices.append(cand)
            return indices

        def __getitem__(self, idx):
            base_idx = self.indices[idx] if self.indices is not None else idx
            indices = self._context_indices(base_idx)
            samples = []
            for i in indices:
                x_i, _ = self.base.get_sample(i)
                samples.append(x_i)
            x = np.stack(samples, axis=0)  # (K, M, F, T)
            x_t = torch.tensor(x.copy(), dtype=torch.float32)
            y_t = torch.tensor(int(self.base.samples[base_idx][1]), dtype=torch.long)
            return x_t, y_t

    # Build dataset and dataloader
    dataset = SleepDataset(args.preprocess_path, args.data_path, frequency=args.sr, segment_len_s=30, num_modalities=1)
    if len(dataset) == 0:
        print("Dataset is empty. Did you download or preprocess data?")
        exit(0)
    # Infer input shapes from a sample
    sample_x, _ = dataset[0]
    # sample_x shape is (K, M, F, T) if temporal enabled, else (M, F, T)
    if sample_x.ndim == 4:
        sample_x0 = sample_x[0]
    else:
        sample_x0 = sample_x
    num_modalities = sample_x0.shape[0]
    freq_bins = sample_x0.shape[1]
    time_frames = sample_x0.shape[2] if sample_x0.ndim >= 3 else 1
    sleepvit_embed_dim = freq_bins * time_frames
    sleepvit_seq_len = num_modalities

    def _pick_num_heads(embed_dim: int) -> int:
        max_heads = min(16, embed_dim)
        for h in range(max_heads, 0, -1):
            if embed_dim % h == 0:
                return h
        return 1

    sleepvit_heads = _pick_num_heads(sleepvit_embed_dim)
    # Infer number of classes from labels if present
    inferred_classes = None
    try:
        labels_all = [int(y) for _, y in dataset.samples]
        if len(labels_all) > 0:
            inferred_classes = int(np.max(labels_all)) + 1
    except Exception:
        pass
    if args.cascade_eval or args.cascade_train:
        num_classes = 5
    elif label_mode == 'binary':
        num_classes = 2
    elif label_mode == 'n1':
        num_classes = 2
    elif label_mode == 'sleep3':
        num_classes = 3
    elif label_mode == 'sleep4':
        num_classes = 4
    elif label_mode == 'full5' and stage_mode == 'fallback5':
        num_classes = 5
    elif label_mode == 'full5':
        num_classes = 5 if args.num_classes <= 0 else args.num_classes
    else:
        num_classes = args.num_classes if args.num_classes > 0 else (inferred_classes if inferred_classes is not None else 5)

    _append_run_log({
        'event': 'dataset_ready',
        'run_id': run_id,
        'timestamp_utc': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
        'dataset_len': len(dataset),
        'num_modalities': num_modalities,
        'freq_bins': freq_bins,
        'time_frames': time_frames,
        'num_classes': num_classes
    })

    def _hash_list(values) -> str:
        h = hashlib.sha256()
        for v in values:
            h.update(str(v).encode('utf-8'))
            h.update(b'\n')
        return h.hexdigest()

    dataset_sample_keys = dataset.sample_keys if hasattr(dataset, 'sample_keys') and len(dataset.sample_keys) == len(dataset) else [str(i) for i in range(len(dataset))]
    dataset_subjects = dataset.sample_subjects if hasattr(dataset, 'sample_subjects') and len(dataset.sample_subjects) == len(dataset) else ['unknown' for _ in range(len(dataset))]
    dataset_signature = _hash_list(dataset_sample_keys)
    print(f"Dataset signature: {dataset_signature[:12]}... ({len(dataset_sample_keys)} samples)")

    split_manifest = None
    split_manifest_path = args.split_manifest_path.strip()
    if split_manifest_path and os.path.isfile(split_manifest_path):
        try:
            with open(split_manifest_path, 'r', encoding='utf-8') as f:
                split_manifest = json.load(f)
            print(f"Loaded split manifest: {split_manifest_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load split manifest {split_manifest_path}: {e}")

    total_len = len(dataset)
    train_set = None
    val_set = None
    test_set = None
    use_subject_split = args.split_mode == 'subject'
    if use_subject_split and (not hasattr(dataset, 'sample_subjects') or len(dataset.sample_subjects) != total_len):
        tqdm.write('Subject metadata unavailable; falling back to per-sample splits.')
        use_subject_split = False

    sample_key_to_idx = {k: i for i, k in enumerate(dataset_sample_keys)}

    def indices_from_keys(keys, split_name: str):
        idxs = []
        missing = []
        for key in keys:
            idx = sample_key_to_idx.get(key)
            if idx is None:
                missing.append(key)
                continue
            idxs.append(idx)
        if missing:
            msg = f"Manifest split '{split_name}' has {len(missing)} missing keys in current dataset"
            if args.strict_split_manifest:
                raise ValueError(msg)
            print('Warning:', msg)
        return idxs

    manifest_train_indices = None
    manifest_val_indices = None
    manifest_test_indices = None
    if split_manifest is not None:
        manifest_mode = split_manifest.get('split_mode')
        if manifest_mode != args.split_mode:
            print(f"Warning: manifest split_mode={manifest_mode} differs from current split_mode={args.split_mode}")
        manifest_sig = split_manifest.get('dataset_signature')
        if manifest_sig and manifest_sig != dataset_signature:
            print('Warning: manifest dataset signature differs from current dataset; using key intersection.')
        train_keys = split_manifest.get('train_keys', [])
        val_keys = split_manifest.get('val_keys', [])
        test_keys = split_manifest.get('test_keys', [])
        manifest_train_indices = indices_from_keys(train_keys, 'train')
        manifest_val_indices = indices_from_keys(val_keys, 'val')
        manifest_test_indices = indices_from_keys(test_keys, 'test')

    if use_subject_split:
        subject_to_indices = defaultdict(list)
        for idx, subj in enumerate(dataset.sample_subjects):
            subject_to_indices[str(subj)].append(idx)
        subject_ids = np.array(list(subject_to_indices.keys()))
        if subject_ids.size == 0:
            raise ValueError('No subjects available for splitting.')
        if split_manifest is not None and 'train_subjects' in split_manifest:
            train_subjects = np.array(split_manifest.get('train_subjects', []), dtype=object)
            val_subjects = np.array(split_manifest.get('val_subjects', []), dtype=object)
            test_subjects = np.array(split_manifest.get('test_subjects', []), dtype=object)
            known_subjects = set(subject_ids.tolist())
            train_subjects = np.array([s for s in train_subjects if s in known_subjects], dtype=object)
            val_subjects = np.array([s for s in val_subjects if s in known_subjects], dtype=object)
            test_subjects = np.array([s for s in test_subjects if s in known_subjects], dtype=object)
            print(f"Using subject splits from manifest -> train: {len(train_subjects)}, val: {len(val_subjects)}, test: {len(test_subjects)}")
        else:
            np.random.shuffle(subject_ids)
            total_subjects = subject_ids.size
            train_subjects_remaining = total_subjects
            val_count = 0
            test_count = 0

            if args.val_split > 0 and train_subjects_remaining > 0:
                desired = max(1, int(args.val_split * total_subjects))
                max_allowed = train_subjects_remaining if args.test_only else max(train_subjects_remaining - 1, 0)
                val_count = min(desired, max_allowed)
                train_subjects_remaining -= val_count
            if args.test_split > 0 and train_subjects_remaining > 0:
                desired = max(1, int(args.test_split * total_subjects))
                max_allowed = train_subjects_remaining if args.test_only else max(train_subjects_remaining - 1, 0)
                test_count = min(desired, max_allowed)
                train_subjects_remaining -= test_count

            if train_subjects_remaining <= 0:
                if args.test_only and (val_count > 0 or test_count > 0):
                    train_subjects_remaining = 0
                else:
                    raise ValueError('Train split length is zero; decrease val/test splits.')

            val_subjects = subject_ids[:val_count]
            test_subjects = subject_ids[val_count:val_count + test_count]
            train_subjects = subject_ids[val_count + test_count:]

        def gather_indices(subject_list):
            idxs = []
            for subject in subject_list:
                idxs.extend(subject_to_indices[str(subject)])
            return idxs

        train_indices = gather_indices(train_subjects)
        val_indices = gather_indices(val_subjects)
        test_indices = gather_indices(test_subjects)
        train_set = torch.utils.data.Subset(dataset, train_indices) if train_indices else None
        val_set = torch.utils.data.Subset(dataset, val_indices) if val_indices else None
        test_set = torch.utils.data.Subset(dataset, test_indices) if test_indices else None
        tqdm.write(f"Subject split counts -> train: {len(train_subjects)}, val: {len(val_subjects)}, test: {len(test_subjects)}")
    else:
        if split_manifest is not None and manifest_train_indices is not None:
            train_indices = manifest_train_indices
            val_indices = manifest_val_indices if manifest_val_indices is not None else []
            test_indices = manifest_test_indices if manifest_test_indices is not None else []
            train_set = torch.utils.data.Subset(dataset, train_indices) if len(train_indices) > 0 else None
            val_set = torch.utils.data.Subset(dataset, val_indices) if len(val_indices) > 0 else None
            test_set = torch.utils.data.Subset(dataset, test_indices) if len(test_indices) > 0 else None
            tqdm.write(f"Using sample splits from manifest -> train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}")
        else:
            val_len = 0
            test_len = 0
            train_len = total_len

            if args.val_split > 0 and train_len > 0:
                desired = max(1, int(args.val_split * total_len))
                max_allowed = train_len if args.test_only else max(train_len - 1, 0)
                val_len = min(desired, max_allowed)
                if val_len > 0:
                    train_len -= val_len
                else:
                    val_len = 0

            if args.test_split > 0 and train_len > 0:
                desired = max(1, int(args.test_split * total_len))
                max_allowed = train_len if args.test_only else max(train_len - 1, 0)
                test_len = min(desired, max_allowed)
                if test_len > 0:
                    train_len -= test_len
                else:
                    test_len = 0

            if train_len <= 0:
                if args.test_only and (val_len > 0 or test_len > 0):
                    train_len = 0
                else:
                    raise ValueError('Train split length is zero; decrease val/test splits.')

            lengths = []
            subset_names = []
            if train_len > 0:
                lengths.append(train_len)
                subset_names.append('train')
            if val_len > 0:
                lengths.append(val_len)
                subset_names.append('val')
            if test_len > 0:
                lengths.append(test_len)
                subset_names.append('test')
            if not lengths:
                raise ValueError('No data available after applying splits.')
            subsets = torch.utils.data.random_split(dataset, lengths)
            split_map = {name: subset for name, subset in zip(subset_names, subsets)}
            train_set = split_map.get('train')
            val_set = split_map.get('val')
            test_set = split_map.get('test')

    def _subset_indices(subset):
        if subset is None:
            return []
        if hasattr(subset, 'indices') and subset.indices is not None:
            return list(subset.indices)
        return list(range(len(subset)))

    train_indices_final = _subset_indices(train_set)
    val_indices_final = _subset_indices(val_set)
    test_indices_final = _subset_indices(test_set)

    if split_manifest_path and args.write_split_manifest and split_manifest is None:
        os.makedirs(os.path.dirname(split_manifest_path) or '.', exist_ok=True)
        manifest_payload = {
            'version': 1,
            'created_utc': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
            'split_mode': args.split_mode,
            'dataset_signature': dataset_signature,
            'dataset_len': len(dataset_sample_keys),
            'data_path': args.data_path,
            'preprocess_path': args.preprocess_path,
            'modalities': args.modalities,
            'train_keys': [dataset_sample_keys[i] for i in train_indices_final],
            'val_keys': [dataset_sample_keys[i] for i in val_indices_final],
            'test_keys': [dataset_sample_keys[i] for i in test_indices_final],
            'train_subjects': sorted(set(dataset_subjects[i] for i in train_indices_final)),
            'val_subjects': sorted(set(dataset_subjects[i] for i in val_indices_final)),
            'test_subjects': sorted(set(dataset_subjects[i] for i in test_indices_final)),
        }
        with open(split_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_payload, f, indent=2)
        print(f"Wrote split manifest: {split_manifest_path}")

    def _resolve_routing_weights_path() -> str:
        if args.routing_weights_path.strip():
            return args.routing_weights_path.strip()
        if args.save_routing_weights:
            return os.path.join(args.save_path, 'routing_weights.json')
        if args.use_routing_weights:
            return os.path.join('result', 'binary', 'routing_weights.json')
        return ''

    routing_weights_path = _resolve_routing_weights_path()
    routing_weights_map = {}
    if args.use_routing_weights:
        if not routing_weights_path or not os.path.isfile(routing_weights_path):
            raise FileNotFoundError(f'Routing weights file not found: {routing_weights_path}')
        with open(routing_weights_path, 'r', encoding='utf-8') as f:
            routing_blob = json.load(f)
        routing_weights_map = routing_blob.get('weights', {}) if isinstance(routing_blob, dict) else {}
        print(f"Loaded routing weights: {routing_weights_path} ({len(routing_weights_map)} keys)")

    def _extract_base_indices(active_set):
        if active_set is None:
            return []
        if isinstance(active_set, TemporalContextDataset):
            if active_set.indices is not None:
                return list(active_set.indices)
            return list(range(len(active_set.base)))
        if hasattr(active_set, 'indices') and active_set.indices is not None:
            return list(active_set.indices)
        return list(range(len(active_set)))
    # Apply temporal context wrapper per split to avoid crossing boundaries
    temporal_radius = max(0, int(args.temporal))
    if temporal_radius > 0:
        if train_set is not None:
            train_set = TemporalContextDataset(dataset, train_set.indices if hasattr(train_set, 'indices') else None, radius=temporal_radius)
        if val_set is not None:
            val_set = TemporalContextDataset(dataset, val_set.indices if hasattr(val_set, 'indices') else None, radius=temporal_radius)
        if test_set is not None:
            test_set = TemporalContextDataset(dataset, test_set.indices if hasattr(test_set, 'indices') else None, radius=temporal_radius)

    # Optional weighted sampler for imbalance
    train_loader = None
    train_eval_loader = None
    train_eval_base_indices = []
    if train_set is not None and len(train_set) > 0:
        base_indices = _extract_base_indices(train_set)
        train_eval_base_indices = list(base_indices)
        train_labels = [int(dataset.samples[i][1]) for i in base_indices]
        needs_routing_weights = bool(args.use_routing_weights and stage_mode in ['n1', 'sleep3', 'sleep4', 'fallback5'])
        sample_weights = None
        if args.use_weighted_sampler:
            counts = np.bincount(train_labels, minlength=num_classes)
            class_w = 1.0 / np.clip(counts, 1, None)
            sample_weights = np.array([class_w[y] for y in train_labels], dtype=np.float32)
        if needs_routing_weights:
            if stage_mode == 'n1':
                route_key = 'n1'
            elif stage_mode == 'sleep3':
                route_key = 'sleep3'
            elif stage_mode == 'sleep4':
                route_key = 'sleep4'
            else:
                route_key = 'fallback'
            route_w = []
            for base_idx in base_indices:
                key = dataset.sample_keys[base_idx] if hasattr(dataset, 'sample_keys') else str(base_idx)
                meta = routing_weights_map.get(key, {}) if isinstance(routing_weights_map, dict) else {}
                w = float(meta.get(route_key, 0.0)) if isinstance(meta, dict) else 0.0
                route_w.append(max(w, 1e-3) ** max(float(args.routing_weight_power), 0.0))
            route_w = np.asarray(route_w, dtype=np.float32)
            sample_weights = route_w if sample_weights is None else (sample_weights * route_w)
            print(f"Applied routing weights for stage={stage_mode} (key={route_key})")
        if sample_weights is not None:
            sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=min(4, args.workers), pin_memory=True)
        else:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=min(4, args.workers), pin_memory=True)
        train_eval_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=min(2, args.workers), pin_memory=True)

    val_loader = None
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=min(2, args.workers), pin_memory=True)

    test_loader = None
    if test_set is not None and len(test_set) > 0:
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=min(2, args.workers), pin_memory=True)

    target_loader = None
    target_micro_loader = None
    target_val_loader = None
    if args.domain_adapt in ['mmd', 'damnet']:
        try:
            target_dataset = SleepDataset(args.target_preprocess_path, args.target_data_path, frequency=args.sr, segment_len_s=30, num_modalities=1, max_preprocessed_files=args.target_max_preprocessed_files)
            if temporal_radius > 0:
                target_dataset = TemporalContextDataset(target_dataset, indices=None, radius=temporal_radius)
            target_bs = args.batch_size if not args.target_batch_size or args.target_batch_size <= 0 else args.target_batch_size
            if args.domain_adapt == 'mmd':
                target_loader = DataLoader(target_dataset, batch_size=target_bs, shuffle=True, num_workers=min(2, args.workers), pin_memory=True)
                tqdm.write(f"Loaded target domain with {len(target_dataset)} samples from {args.target_preprocess_path}")
            else:
                total_target = len(target_dataset)
                if args.micro_label_count and args.micro_label_count > 0:
                    micro_len = min(args.micro_label_count, total_target)
                else:
                    micro_len = max(1, int(args.micro_label_ratio * total_target)) if total_target > 0 else 0
                val_len = max(0, total_target - micro_len)
                if micro_len == 0 or val_len == 0:
                    raise ValueError('DAM-Net requires both micro-labeled and validation target splits; adjust micro_label_ratio/count.')
                target_micro, target_val = torch.utils.data.random_split(target_dataset, [micro_len, val_len])
                target_micro_loader = DataLoader(target_micro, batch_size=target_bs, shuffle=True, num_workers=min(2, args.workers), pin_memory=True)
                target_val_loader = DataLoader(target_val, batch_size=target_bs, shuffle=False, num_workers=min(2, args.workers), pin_memory=True)
                tqdm.write(f"Loaded target domain with {len(target_dataset)} samples; micro-labeled {micro_len}, val {val_len} from {args.target_preprocess_path}")
        except Exception as e:
            print('Warning: could not load target domain for MMD:', e)
            target_loader = None
            target_micro_loader = None
            target_val_loader = None
    if args.domain_adapt == 'mmd' and target_loader is None:
        print('Warning: MMD enabled but no target loader available; training will be source-only.')
    if args.domain_adapt == 'damnet' and (target_micro_loader is None or target_val_loader is None):
        print('Warning: DAM-Net enabled but target splits unavailable; training will be source-only.')

    if not args.test_only and not args.cascade_train and train_loader is None:
        raise ValueError('Training split is empty; reduce val/test splits or disable test_only.')

    if val_loader is None and train_loader is not None:
        tqdm.write('Warning: validation split empty; using training data for validation metrics.')
        val_loader = train_loader

    def format_batch_for_model(x_batch: torch.Tensor) -> torch.Tensor:
        if args.model == 'sleepvit':
            if x_batch.ndim != 4:
                raise ValueError('SleepViT expects inputs of shape (B, M, F, T).')
            B, M, F, T = x_batch.shape
            return x_batch.contiguous().view(B, M, F * T)
        return x_batch

    def should_collect_energy() -> bool:
        return bool(args.track_energy or args.energy_threshold is not None)

    def compute_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        scaled = logits / max(temperature, 1e-6)
        return -temperature * torch.logsumexp(scaled, dim=1)

    def _parse_sigmas(sigmas_str: str):
        vals = []
        for s in sigmas_str.split(','):
            s = s.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except Exception:
                continue
        return vals if vals else [1.0]

    def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigmas):
        # x: (N, D), y: (M, D)
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        dist = x_norm - 2 * x @ y.t() + y_norm.t()
        kernels = 0.0
        for sigma in sigmas:
            gamma = 1.0 / max(2.0 * sigma * sigma, 1e-6)
            kernels = kernels + torch.exp(-gamma * dist)
        return kernels / max(len(sigmas), 1)

    def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigmas):
        if x.size(0) == 0 or y.size(0) == 0:
            return torch.tensor(0.0, device=x.device)
        k_xx = gaussian_kernel(x, x, sigmas)
        k_yy = gaussian_kernel(y, y, sigmas)
        k_xy = gaussian_kernel(x, y, sigmas)
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    def classwise_mmd_loss(x_s: torch.Tensor, y_s: torch.Tensor, x_t: torch.Tensor, y_t: torch.Tensor, num_classes: int, sigmas):
        total = 0.0
        count = 0
        for c in range(num_classes):
            xs = x_s[y_s == c]
            xt = x_t[y_t == c]
            if xs.size(0) == 0 or xt.size(0) == 0:
                continue
            total = total + mmd_loss(xs, xt, sigmas)
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=x_s.device)
        return total / count

    def supervised_contrastive_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07,
        class_weight_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1, 1)
        if labels.size(0) <= 1:
            return torch.tensor(0.0, device=features.device)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits = torch.div(torch.matmul(features, features.T), max(temperature, 1e-6))
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos
        if class_weight_tensor is not None:
            label_idx = labels.view(-1).long()
            safe_weights = class_weight_tensor.to(features.device)
            sample_weights = safe_weights[label_idx]
            denom = torch.clamp(sample_weights.sum(), min=1e-12)
            return (loss * sample_weights).sum() / denom
        return loss.mean()

    def _compute_energy_level_thresholds(energies_np: np.ndarray, levels: List[float]) -> Dict[float, float]:
        out: Dict[float, float] = {}
        if energies_np.size == 0:
            return out
        for level in levels:
            cov = float(level) / 100.0
            cov = min(1.0, max(0.0, cov))
            out[float(level)] = float(np.quantile(energies_np, cov))
        return out

    def _compute_coverage_from_thresholds(energies_np: np.ndarray, thresholds: Dict[float, float]) -> Dict[float, float]:
        out: Dict[float, float] = {}
        if energies_np.size == 0:
            return out
        total = float(max(1, energies_np.size))
        for lvl, thr in thresholds.items():
            out[float(lvl)] = float(np.sum(energies_np <= float(thr)) / total)
        return out

    def log_uncertainty_metrics(
        split_name: str,
        details,
        epoch=None,
        level_thresholds: Optional[Dict[float, float]] = None,
        level_reference_coverages: Optional[Dict[float, float]] = None,
    ):
        if details is None:
            return
        preds_np = details['preds'].cpu().numpy()
        targets_np = details['targets'].cpu().numpy()
        energies_np = details['energies'].cpu().numpy()

        def _metrics_from_arrays(pred_arr: np.ndarray, targ_arr: np.ndarray):
            if pred_arr.size == 0:
                return float('nan'), float('nan'), float('nan'), float('nan'), [float('nan')] * num_classes
            cm = np.zeros((num_classes, num_classes), dtype=np.int64)
            for pred_val, targ_val in zip(pred_arr, targ_arr):
                cm[int(targ_val), int(pred_val)] += 1
            total = int(cm.sum())
            acc = float(np.trace(cm)) / max(1, total)
            f1_list = []
            for class_idx in range(num_classes):
                tp = cm[class_idx, class_idx]
                fp = cm[:, class_idx].sum() - tp
                fn = cm[class_idx, :].sum() - tp
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1_val = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
                f1_list.append(f1_val)
            f1_macro = float(np.mean(f1_list)) if f1_list else float('nan')
            tp_sum = int(np.trace(cm))
            fp_sum = int(cm.sum(axis=0).sum() - tp_sum)
            fn_sum = int(cm.sum(axis=1).sum() - tp_sum)
            denom = (2 * tp_sum + fp_sum + fn_sum)
            f1_micro = 0.0 if denom == 0 else (2 * tp_sum) / denom
            po = float(np.trace(cm)) / max(1, total)
            pe = float(np.sum(cm.sum(axis=0) * cm.sum(axis=1))) / max(1, total * total)
            kappa = 0.0 if (1 - pe) == 0 else (po - pe) / (1 - pe)
            row_sums = cm.sum(axis=1)
            per_class_acc = [float(cm[i, i] / row_sums[i]) if row_sums[i] > 0 else float('nan') for i in range(num_classes)]
            return acc, f1_macro, f1_micro, kappa, per_class_acc

        def _fmt_metric(value: float) -> str:
            return 'nan' if not np.isfinite(value) else f'{value:.3f}'

        if label_mode == 'binary' and num_classes == 2:
            class_names = ['W', 'S']
        elif label_mode == 'n1' and num_classes == 2:
            class_names = ['nonN1', 'N1']
        elif label_mode == 'sleep3' and num_classes == 3:
            class_names = ['N2', 'N3', 'REM']
        elif label_mode == 'sleep4' and num_classes == 4:
            class_names = ['N1', 'N2', 'N3', 'REM']
        elif num_classes == 5:
            class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        else:
            class_names = [f'C{i}' for i in range(num_classes)]
        if args.energy_threshold is not None:
            mask = energies_np <= args.energy_threshold
            kept = int(mask.sum())
            coverage = kept / max(1, len(energies_np))
            if kept > 0:
                filtered_acc = float((preds_np[mask] == targets_np[mask]).mean())
                print(f"[{split_name}] energy-threshold acc={filtered_acc:.3f} coverage={coverage:.3f} (threshold {args.energy_threshold})")
            else:
                print(f"[{split_name}] energy-threshold acc=nan coverage=0.000 (threshold {args.energy_threshold}) - no samples kept")
        levels_to_report = list(energy_levels)
        if not levels_to_report and level_thresholds is not None:
            try:
                levels_to_report = sorted([float(k) for k in level_thresholds.keys()], reverse=True)
            except Exception:
                levels_to_report = []

        if levels_to_report:
            total = len(energies_np)
            if total == 0:
                return
            level_rows = []
            for level in levels_to_report:
                if level_thresholds is not None and float(level) in level_thresholds:
                    thr = float(level_thresholds[float(level)])
                else:
                    target_cov = level / 100.0
                    thr = float(np.quantile(energies_np, target_cov))
                mask = energies_np <= thr
                kept = int(mask.sum())
                coverage = kept / max(1, total)
                if kept > 0:
                    acc = float((preds_np[mask] == targets_np[mask]).mean())
                    acc_lvl, f1_macro_lvl, f1_micro_lvl, kappa_lvl, per_class_lvl = _metrics_from_arrays(
                        preds_np[mask],
                        targets_np[mask]
                    )
                    print(
                        f"[{split_name}] coverage {level:.1f}% -> energy<= {thr:.4f} acc={acc:.3f} (actual {coverage:.3f})"
                    )
                    level_rows.append({
                        'level': level,
                        'energy': thr,
                        'coverage': coverage,
                        'reference_coverage': (
                            float(level_reference_coverages[float(level)])
                            if level_reference_coverages is not None and float(level) in level_reference_coverages
                            else float('nan')
                        ),
                        'n': kept,
                        'acc': acc_lvl,
                        'f1_macro': f1_macro_lvl,
                        'f1_micro': f1_micro_lvl,
                        'kappa': kappa_lvl,
                        'per_class': per_class_lvl,
                    })
                else:
                    print(
                        f"[{split_name}] coverage {level:.1f}% -> energy<= {thr:.4f} acc=nan (actual {coverage:.3f})"
                    )
                    level_rows.append({
                        'level': level,
                        'energy': thr,
                        'coverage': coverage,
                        'reference_coverage': (
                            float(level_reference_coverages[float(level)])
                            if level_reference_coverages is not None and float(level) in level_reference_coverages
                            else float('nan')
                        ),
                        'n': kept,
                        'acc': float('nan'),
                        'f1_macro': float('nan'),
                        'f1_micro': float('nan'),
                        'kappa': float('nan'),
                        'per_class': [float('nan')] * num_classes,
                    })

            if args.track_energy and args.test_only and level_rows:
                include_ref_cov = any(np.isfinite(float(r.get('reference_coverage', float('nan')))) for r in level_rows)
                base_cols = ['level%', 'energy<=', 'coverage']
                if include_ref_cov:
                    base_cols.extend(['val_cov', 'delta_cov'])
                base_cols.extend(['n', 'acc', 'f1_macro', 'f1_micro', 'kappa'])
                class_cols = [f'acc_{name}' for name in class_names]
                headers = base_cols + class_cols
                rows = []
                for row in level_rows:
                    ref_cov = float(row.get('reference_coverage', float('nan')))
                    delta_cov = row['coverage'] - ref_cov if np.isfinite(ref_cov) else float('nan')
                    row_vals = [
                        f"{row['level']:.1f}",
                        f"{row['energy']:.4f}",
                        f"{row['coverage']:.3f}",
                    ]
                    if include_ref_cov:
                        row_vals.extend([_fmt_metric(ref_cov), _fmt_metric(delta_cov)])
                    row_vals.extend([
                        str(row['n']),
                        _fmt_metric(row['acc']),
                        _fmt_metric(row['f1_macro']),
                        _fmt_metric(row['f1_micro']),
                        _fmt_metric(row['kappa']),
                    ])
                    row_vals.extend(_fmt_metric(v) for v in row['per_class'])
                    rows.append(row_vals)

                col_widths = [len(h) for h in headers]
                for row in rows:
                    for idx, val in enumerate(row):
                        col_widths[idx] = max(col_widths[idx], len(val))

                sep = ' | '
                header_line = sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
                divider = '-+-'.join('-' * col_widths[i] for i in range(len(headers)))
                print(f"[{split_name}] Energy levels metrics table:")
                print(header_line)
                print(divider)
                for row in rows:
                    print(sep.join(row[i].ljust(col_widths[i]) for i in range(len(headers))))

                if include_ref_cov:
                    print(f"[{split_name}] Coverage comparison vs validation thresholds:")
                    for row in level_rows:
                        ref_cov = float(row.get('reference_coverage', float('nan')))
                        if np.isfinite(ref_cov):
                            delta_cov = row['coverage'] - ref_cov
                            print(
                                f"[{split_name}] level {row['level']:.1f}%: val_cov={ref_cov:.3f} test_cov={row['coverage']:.3f} delta={delta_cov:+.3f}"
                            )

                def _map_per_class_to_full5(per_class_vals, names):
                    mapped = {
                        'Wake': float('nan'),
                        'N1': float('nan'),
                        'N2': float('nan'),
                        'N3': float('nan'),
                        'Rem': float('nan'),
                    }
                    for cls_name, cls_val in zip(names, per_class_vals):
                        key = str(cls_name).strip().upper()
                        if key in ['W', 'WAKE']:
                            mapped['Wake'] = cls_val
                        elif key == 'N1':
                            mapped['N1'] = cls_val
                        elif key == 'N2':
                            mapped['N2'] = cls_val
                        elif key == 'N3':
                            mapped['N3'] = cls_val
                        elif key in ['R', 'REM']:
                            mapped['Rem'] = cls_val
                    return mapped

                print(f"[{split_name}] LaTeX rows:")
                print("Model & accuracy & Macro F1 & kappa & Wake & N1 & N2 & N3 & Rem\\\\")
                for row in level_rows:
                    mapped = _map_per_class_to_full5(row['per_class'], class_names)
                    model_tag = f"{args.model}_{split_name}_L{int(round(row['level']))}"
                    latex_row = (
                        f"{model_tag} & {_fmt_metric(row['acc'])} & {_fmt_metric(row['f1_macro'])} & {_fmt_metric(row['kappa'])} "
                        f"& {_fmt_metric(mapped['Wake'])} & {_fmt_metric(mapped['N1'])} & {_fmt_metric(mapped['N2'])} "
                        f"& {_fmt_metric(mapped['N3'])} & {_fmt_metric(mapped['Rem'])}\\\\"
                    )
                    print(latex_row)
        if args.track_energy:
            tag = f"{split_name}_epoch{epoch}" if epoch is not None else f"{split_name}_final"
            os.makedirs(args.save_path, exist_ok=True)
            stacked = np.column_stack((targets_np, preds_np, energies_np))
            np.save(os.path.join(args.save_path, f'energy_{tag}.npy'), stacked)

    def plot_energy_accuracy_curve(details, split_name: str, epoch: int):
        if details is None:
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        preds_np = details['preds'].cpu().numpy()
        targets_np = details['targets'].cpu().numpy()
        energies_np = details['energies'].cpu().numpy()
        if energies_np.size == 0:
            return
        e_min = float(np.min(energies_np))
        e_max = float(np.max(energies_np))
        if not np.isfinite(e_min) or not np.isfinite(e_max):
            return
        if e_max <= e_min:
            e_max = e_min + 1e-6

        # Dense thresholds for smooth curves (replaces coarse step-based sampling)
        n_points = int(min(400, max(120, energies_np.size)))
        thresholds = np.linspace(e_min, e_max, n_points, dtype=np.float64)

        accs = []
        covs = []
        total = len(energies_np)
        for t in thresholds:
            mask = energies_np <= t
            kept = int(mask.sum())
            cov = kept / max(1, total)
            if kept > 0:
                acc = float((preds_np[mask] == targets_np[mask]).mean())
            else:
                acc = float('nan')
            accs.append(acc)
            covs.append(cov)

        thresholds = np.asarray(thresholds, dtype=np.float64)
        accs = np.asarray(accs, dtype=np.float64)
        covs = np.asarray(covs, dtype=np.float64)

        # Drop unstable extreme ends (very low/high coverage) to avoid tail anomalies
        core_mask = (covs >= 0.01) & (covs <= 0.99) & np.isfinite(accs)
        if int(np.sum(core_mask)) >= 20:
            thresholds = thresholds[core_mask]
            accs = accs[core_mask]
            covs = covs[core_mask]

        # Edge-aware smoothing for visual quality without boundary cliff artifacts
        def _smooth_edge(arr: np.ndarray, window: int = 9) -> np.ndarray:
            if arr.size == 0:
                return arr
            w = int(max(3, window))
            if w % 2 == 0:
                w += 1
            pad = w // 2
            kernel = np.ones(w, dtype=np.float64) / float(w)
            arr_clean = np.nan_to_num(arr, nan=np.nanmean(arr) if np.isfinite(np.nanmean(arr)) else 0.0)
            arr_pad = np.pad(arr_clean, (pad, pad), mode='edge')
            return np.convolve(arr_pad, kernel, mode='valid')

        win = 9
        accs_smooth = _smooth_edge(accs, window=win)
        covs_smooth = _smooth_edge(covs, window=win)

        # Explicit +3 font size bump for energy visualization text
        def _font_size_points(rc_key: str, fallback: float) -> float:
            raw_value = plt.rcParams.get(rc_key, fallback)
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                try:
                    from matplotlib.font_manager import FontProperties  # type: ignore
                    return float(FontProperties(size=raw_value).get_size_in_points())
                except Exception:
                    return float(fallback)

        base_label_fs = _font_size_points('axes.labelsize', 10.0)
        base_title_fs = _font_size_points('axes.titlesize', 12.0)
        base_tick_fs = _font_size_points('xtick.labelsize', 10.0)
        base_legend_fs = _font_size_points('legend.fontsize', 10.0)
        label_fs = base_label_fs + 3.0
        title_fs = base_title_fs + 3.0
        tick_fs = base_tick_fs + 3.0
        legend_fs = base_legend_fs + 3.0

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(thresholds, accs_smooth, linewidth=2.0, label='Accuracy (smoothed)')
        ax1.set_xlabel('Energy threshold (keep if < threshold)', fontsize=label_fs)
        ax1.set_ylabel('Accuracy', fontsize=label_fs)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=tick_fs)

        # 4th-degree polynomial fit for confidence estimation equation: accuracy = f(energy)
        fit_mask = np.isfinite(accs_smooth)
        if int(np.sum(fit_mask)) >= 6:
            poly = np.polyfit(thresholds[fit_mask], accs_smooth[fit_mask], deg=4)
            acc_fit = np.polyval(poly, thresholds)
            ax1.plot(thresholds, acc_fit, linestyle='--', linewidth=1.8, color='crimson', label='4th-degree fit')

        ax2 = ax1.twinx()
        ax2.plot(thresholds, covs_smooth, color='orange', linewidth=1.8, label='Coverage (smoothed)')
        ax2.set_ylabel('Coverage', fontsize=label_fs)
        ax2.tick_params(axis='y', labelsize=tick_fs)
        title = f"Energy Accuracy/Coverage ({split_name}) epoch {epoch}"
        ax1.set_title(title, fontsize=title_fs)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=legend_fs)
        os.makedirs(args.save_path, exist_ok=True)
        fig.savefig(os.path.join(args.save_path, f'energy_curve_{split_name}_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def visualize_tsne(loader, split_name: str, epoch: Optional[int] = None):
        try:
            from sklearn.manifold import TSNE  # type: ignore
        except Exception:
            print('Warning: scikit-learn not available; skipping t-SNE visualization.')
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            print('Warning: matplotlib not available; skipping t-SNE visualization.')
            return
        model.eval()
        if temporal_head is not None:
            temporal_head.eval()
        feats = []
        labels = []
        max_samples = max(100, int(args.tsne_max_samples))
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                if temporal_radius > 0:
                    B, K, M, F, T = x.shape
                    x = x[:, temporal_radius, ...]
                x = format_batch_for_model(x).to(device)
                logits = model(x)
                if isinstance(logits, (list, tuple)):
                    logits = torch.stack(logits, dim=0).mean(dim=0)
                feats.append(logits.detach().cpu())
                labels.append(y.detach().cpu())
                if sum(f.shape[0] for f in feats) >= max_samples:
                    break
        if not feats:
            return
        X = torch.cat(feats, dim=0)[:max_samples].numpy()
        y_all = torch.cat(labels, dim=0)[:max_samples].numpy()
        tsne = TSNE(n_components=2, perplexity=float(args.tsne_perplexity), init='pca', learning_rate='auto')
        X_2d = tsne.fit_transform(X)
        if label_mode == 'binary':
            class_names = ['Wake', 'Sleep']
            palette = ['#1f77b4', '#d62728']
        elif label_mode == 'n1':
            class_names = ['non-N1', 'N1']
            palette = ['#1f77b4', '#d62728']
        elif label_mode == 'sleep3':
            class_names = ['N2', 'N3', 'REM']
            palette = ['#ff7f0e', '#2ca02c', '#d62728']
        elif label_mode == 'sleep4':
            class_names = ['N1', 'N2', 'N3', 'REM']
            palette = ['#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
        else:
            class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
            palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
        try:
            from matplotlib.colors import ListedColormap  # type: ignore
        except Exception:
            ListedColormap = None
        cmap = ListedColormap(palette[:len(class_names)]) if ListedColormap is not None else 'tab10'
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_all, cmap=cmap, s=10, alpha=0.75)
        ax.set_title(f"t-SNE Features ({split_name})" + (f" epoch {epoch}" if epoch is not None else ""))
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(len(class_names)))
        cbar.ax.set_yticklabels(class_names)
        os.makedirs(args.save_path, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else ""
        fig.savefig(os.path.join(args.save_path, f'tsne_{split_name}{suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Additional class-separation view:
        # project features to the most discriminative 1D axis (from PCA components)
        # and plot smoothed class histograms along that axis.
        X_center = X - X.mean(axis=0, keepdims=True)
        axis_values = None
        axis_label = 'Feature 1'
        fisher_best = None
        try:
            from sklearn.decomposition import PCA  # type: ignore
            n_comp = int(min(10, X_center.shape[0], X_center.shape[1]))
            if n_comp >= 1:
                pca = PCA(n_components=n_comp)
                Z = pca.fit_transform(X_center)
                best_idx = 0
                best_score = -np.inf
                for comp_idx in range(Z.shape[1]):
                    z = Z[:, comp_idx]
                    global_mean = float(np.mean(z))
                    between = 0.0
                    within = 0.0
                    for cls in np.unique(y_all):
                        cls_mask = y_all == cls
                        if not np.any(cls_mask):
                            continue
                        z_cls = z[cls_mask]
                        n_cls = z_cls.size
                        mu_cls = float(np.mean(z_cls))
                        var_cls = float(np.var(z_cls))
                        between += n_cls * (mu_cls - global_mean) ** 2
                        within += n_cls * var_cls
                    score = between / max(within, 1e-12)
                    if score > best_score:
                        best_score = score
                        best_idx = comp_idx
                axis_values = Z[:, best_idx]
                fisher_best = float(best_score)
                axis_label = f'PCR axis (PC{best_idx + 1})'
        except Exception:
            axis_values = None

        if axis_values is None:
            # fallback: first centered feature if PCA is unavailable
            axis_values = X_center[:, 0]

        fig_h, ax_h = plt.subplots(figsize=(7, 4))
        uniq_classes = sorted(np.unique(y_all).tolist())
        if axis_values.size > 1:
            span_min = float(np.min(axis_values))
            span_max = float(np.max(axis_values))
            if span_max <= span_min:
                span_max = span_min + 1e-6
        else:
            span_min, span_max = -1.0, 1.0
        bins = np.linspace(span_min, span_max, 70)
        centers = 0.5 * (bins[:-1] + bins[1:])
        sigma_bins = 1.6
        rad = int(max(3, np.ceil(3 * sigma_bins)))
        xk = np.arange(-rad, rad + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (xk / sigma_bins) ** 2)
        kernel /= np.sum(kernel)

        for cls in uniq_classes:
            cls = int(cls)
            cls_mask = y_all == cls
            if not np.any(cls_mask):
                continue
            z_cls = axis_values[cls_mask]
            hist, _ = np.histogram(z_cls, bins=bins, density=True)
            smooth = np.convolve(hist, kernel, mode='same')
            label_name = class_names[cls] if 0 <= cls < len(class_names) else f'Class {cls}'
            color = palette[cls % len(palette)]
            ax_h.plot(centers, smooth, linewidth=1.8, color=color, label=label_name)
            ax_h.fill_between(centers, smooth, alpha=0.18, color=color)

        title = f"Class separation ({split_name})" + (f" epoch {epoch}" if epoch is not None else "")
        if fisher_best is not None:
            title += f" | Fisher={fisher_best:.3f}"
        ax_h.set_title(title)
        ax_h.set_xlabel(axis_label)
        ax_h.set_ylabel('Smoothed density')
        ax_h.grid(alpha=0.25)
        ax_h.legend(loc='best', fontsize=9)
        fig_h.savefig(os.path.join(args.save_path, f'class_hist_{split_name}{suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_h)

    print(args.model)
    def build_convnext(num_classes: int, size: str):
        if size == 'tiny':
            from torchvision.models import convnext_tiny
            return convnext_tiny(progress=True, num_classes=num_classes)
        if size == 'small':
            from torchvision.models import convnext_small
            return convnext_small(progress=True, num_classes=num_classes)
        from torchvision.models import convnext_large
        return convnext_large(progress=True, num_classes=num_classes)

    def adjust_convnext_input(model_local: nn.Module, in_channels: int) -> None:
        if in_channels == 3:
            return
        try:
            first_conv = model_local.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                dilation=first_conv.dilation,
                groups=first_conv.groups,
                bias=first_conv.bias is not None,
            )
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)
            model_local.features[0][0] = new_conv
            print(f"Adjusted convnext input channels: {in_channels} -> {first_conv.out_channels}")
        except Exception as e:
            print('Warning: could not adjust convnext input channels:', e)
    if args.model == 'simple_cnn':
        # lazy import to avoid requiring the file unless requested
        from models.simple_cnn import SimpleSleepCNN
        model = SimpleSleepCNN(num_modalities=num_modalities, freq_bins=freq_bins, num_classes=num_classes,
                               base_channels=32, dropout=0.3).to(device)
    elif args.model == 'transformer':
        model = SleepStageClassifier(num_modalities=num_modalities, num_classes=num_classes, freq_bins=freq_bins,
                                    dropout=0.1).to(device)
    elif args.model in ['resnet', 'resnet152']:
        # lazy import to avoid requiring the file unless requested
        print("Using resnet152")
        from torchvision.models import resnet152
        model = resnet152(progress=True, num_classes=num_classes).to(device)
        if num_modalities != 3:
            try:
                first_conv = model.conv1
                new_conv = nn.Conv2d(
                    num_modalities,
                    first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    dilation=first_conv.dilation,
                    groups=first_conv.groups,
                    bias=first_conv.bias is not None,
                )
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
                model.conv1 = new_conv
                print(f"Adjusted resnet input channels: {num_modalities} -> {first_conv.out_channels}")
            except Exception as e:
                print('Warning: could not adjust resnet input channels:', e)
    elif args.model == 'convnext':
        # lazy import to avoid requiring the file unless requested
        print(f"Using convnext ({args.convnext_size})")
        model = build_convnext(num_classes=num_classes, size=args.convnext_size).to(device)
        adjust_convnext_input(model, num_modalities)
    elif args.model == 'sleepvit':
        print(f"Configuring SleepViT with embed_dim={sleepvit_embed_dim}, seq_len={sleepvit_seq_len}, heads={sleepvit_heads}")
        model = InterEpochTransformer(
            num_classes=num_classes,
            embed_dim=sleepvit_embed_dim,
            depth=4,
            num_heads=sleepvit_heads,
            num_seq=sleepvit_seq_len,
            mlp_ratio=4.0
        ).to(device)
    feature_extractor = None
    temporal_feature_dim = None
    if args.model == 'convnext':
        feature_extractor = create_feature_extractor(model, return_nodes={'avgpool': 'feat'}).to(device)
        with torch.no_grad():
            sample_feat = torch.tensor(sample_x0[None, ...], dtype=torch.float32, device=device)
            sample_feat = format_batch_for_model(sample_feat)
            feat_out = feature_extractor(sample_feat)['feat']
            temporal_feature_dim = int(feat_out.view(feat_out.size(0), -1).size(1))
    print(model.__class__.__name__, 'built with', num_modalities, 'modalities and', freq_bins, 'freq bins; classes =', num_classes)

    _append_run_log({
        'event': 'model_ready',
        'run_id': run_id,
        'timestamp_utc': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
        'model': model.__class__.__name__,
        'temporal_radius': temporal_radius,
        'temporal_only': bool(args.temporal_only),
        'save_path': args.save_path,
        'log_dir': args.log_dir
    })

    # Class weights from dataset to mitigate imbalance
    class_weights = None
    class_counts = None
    try:
        counts = np.bincount([int(y) for _, y in dataset.samples], minlength=num_classes)
        class_counts = counts
        weights = 1.0 / np.clip(counts, 1, None)
        weights = weights * (num_classes / np.sum(weights))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print('Class counts:', counts.tolist())
        print('Class weights:', [round(float(w), 4) for w in weights])
    except Exception as e:
        raise RuntimeError('Could not compute class weights:', e)
    use_binary_loss = bool(label_mode == 'binary')
    bce_loss = None
    if use_binary_loss:
        if num_classes != 2:
            raise ValueError(f'binary stage expects 2 classes, got {num_classes}')
        # pos_weight balances positives vs negatives for BCEWithLogits
        if class_counts is not None and len(class_counts) >= 2:
            neg = float(class_counts[0])
            pos = float(class_counts[1])
            pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = None
        print(f"Using BCEWithLogitsLoss for binary wake/sleep (pos_weight={float(pos_weight.item()):.4f})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if use_binary_loss:
            if logits.dim() == 2 and logits.size(1) == 2:
                logit_pos = logits[:, 1] - logits[:, 0]
            elif logits.dim() == 1:
                logit_pos = logits
            elif logits.dim() == 2 and logits.size(1) == 1:
                logit_pos = logits.view(-1)
            else:
                raise ValueError(f'Unexpected logits shape for binary loss: {tuple(logits.shape)}')
            targets_f = targets.float()
            return bce_loss(logit_pos, targets_f)  # type: ignore[operator]
        return criterion(logits, targets)  # type: ignore[operator]
    temporal_head = None
    if temporal_radius > 0:
        input_dim = (2 * temporal_radius + 1) * (temporal_feature_dim or num_classes)
        temporal_head = TemporalHead(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=args.temporal_hidden,
            dropout=args.temporal_dropout
        ).to(device)
        print(f"Temporal head enabled: radius={temporal_radius}, input={input_dim} -> {num_classes} | hidden={args.temporal_hidden}")
    optimizer_base = torch.optim.AdamW(list(model.parameters()), lr=args.lr, weight_decay=1e-2)
    optimizer_temporal = None
    if temporal_head is not None:
        optimizer_temporal = torch.optim.AdamW(list(temporal_head.parameters()), lr=args.lr, weight_decay=1e-2)

    def build_lr_scheduler(optimizer, total_epochs: int):
        if optimizer is None or args.lr_scheduler == 'none':
            return None
        if args.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(total_epochs)),
                eta_min=float(args.lr_min),
            )
        if args.lr_scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, int(args.lr_step_size)),
                gamma=float(args.lr_gamma),
            )
        return None

    scheduler_base = build_lr_scheduler(optimizer_base, args.epochs)
    scheduler_temporal = build_lr_scheduler(optimizer_temporal, args.epochs)

    mmd_enabled = args.domain_adapt == 'mmd' and not bool(args.temporal_only)
    mmd_sigmas = _parse_sigmas(args.mmd_sigmas)

    temporal_only = bool(args.temporal_only)
    if temporal_only and temporal_radius <= 0:
        raise ValueError('temporal_only requires --temporal > 0')
    if temporal_only:
        if not os.path.isfile(args.base_checkpoint):
            raise FileNotFoundError(f'Base checkpoint not found: {args.base_checkpoint}')
        base_ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
        if 'model' not in base_ckpt:
            raise KeyError('Base checkpoint missing "model" key')
        model.load_state_dict(base_ckpt['model'])
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        if temporal_head is None:
            raise ValueError('Temporal head not initialized; set --temporal > 0')
        if os.path.isfile(args.temporal_checkpoint):
            tmp_ckpt = torch.load(args.temporal_checkpoint, map_location=device, weights_only=False)
            if 'temporal_head' in tmp_ckpt:
                temporal_head.load_state_dict(tmp_ckpt['temporal_head'])
            else:
                print('Warning: temporal checkpoint missing "temporal_head" key; training from scratch.')

    def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for p, t in zip(preds_np, targets_np):
            cm[t, p] += 1
        total = cm.sum()
        acc = float(np.trace(cm)) / max(1, total)
        f1s = []
        for c in range(num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
        po = float(np.trace(cm)) / max(1, total)
        pe = float(np.sum(cm.sum(axis=0) * cm.sum(axis=1))) / max(1, total * total)
        kappa = 0.0 if (1 - pe) == 0 else (po - pe) / (1 - pe)
        return acc, macro_f1, kappa, cm

    def plot_and_save_cm(cm: np.ndarray, class_names, save_path: str, writer=None, tag: str = 'ConfusionMatrix/val', step: int = 0):
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=class_names, yticklabels=class_names, ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, np.clip(row_sums, 1, None), dtype=np.float64) * 100.0
        thresh = cm.max() / 2.0 if cm.size > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = cm_pct[i, j]
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n{pct:.1f}%",
                    ha='center',
                    va='center',
                    color='white' if cm[i, j] > thresh else 'black'
                )
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        if writer is not None:
            writer.add_figure(tag, fig, global_step=step)
        plt.close(fig)

    def run_epoch(loader, train=True, collect_energy=False, target_loader=None, optim_base=None, optim_temp=None, use_mmd=True):
        if temporal_only:
            model.eval()
        else:
            model.train(train)
        if temporal_head is not None:
            temporal_head.train(train)
        total_loss = 0.0
        total_base_loss = 0.0
        total_temporal_loss = 0.0
        total_mmd_loss = 0.0
        total_correct = 0
        total = 0
        all_preds = []
        all_targets = []
        energy_records = [] if collect_energy else None
        batch_idx = 0
        target_iter = None
        mmd_active = bool(mmd_enabled and use_mmd)
        if train and mmd_active and target_loader is not None:
            target_iter = iter(target_loader)
        loadbar = tqdm(total=len(loader), desc="Processing batches", position=1)
        for batch in loader:
            x, y = batch
            # sanitize batch inputs to avoid propagating NaNs/Infs
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if temporal_radius > 0:
                # x shape: (B, K, M, F, T)
                B, K, M, F, T = x.shape
                x_full = x
                center_x = x_full[:, temporal_radius, ...]
                center_x = format_batch_for_model(center_x).to(device)
            else:
                x_full = None
                center_x = format_batch_for_model(x).to(device)
            y = y.to(device)
            if train and not temporal_only:
                (optim_base or optimizer_base).zero_grad(set_to_none=True)
            with torch.no_grad() if temporal_only else torch.enable_grad():
                center_logits = model(center_x)
            if isinstance(center_logits, (list, tuple)):
                center_logits = torch.stack(center_logits, dim=0).mean(dim=0)
            if feature_extractor is not None:
                center_feats = feature_extractor(center_x)['feat']
                center_feats = center_feats.view(center_feats.size(0), -1)
            else:
                center_feats = center_logits
            temporal_logits = None
            if temporal_radius > 0 and temporal_head is not None and x_full is not None:
                x_full = x_full.view(B * K, M, F, T)
                x_full = format_batch_for_model(x_full).to(device)
                with torch.no_grad():
                    if feature_extractor is not None:
                        feats_full = feature_extractor(x_full)['feat']
                        feats_full = feats_full.view(feats_full.size(0), -1)
                        feats_full = feats_full.view(B, K, -1)
                    else:
                        logits_full = model(x_full)
                        if isinstance(logits_full, (list, tuple)):
                            logits_full = torch.stack(logits_full, dim=0).mean(dim=0)
                        if logits_full.ndim != 2:
                            raise ValueError('Model logits expected shape (B*K, C) for temporal mode.')
                        feats_full = logits_full.view(B, K, -1)
                temporal_logits = temporal_head(feats_full.reshape(B, K * feats_full.size(-1)))
            if torch.isnan(center_logits).any() or torch.isinf(center_logits).any():
                print('Warning: logits contain NaN/Inf; skipping batch')
                continue
            final_logits = temporal_logits if temporal_logits is not None else center_logits
            if collect_energy and energy_records is not None:
                energy_records.append(torch.nan_to_num(compute_energy(final_logits.detach())).cpu())
            mmd_val = None
            if train and mmd_active and target_loader is not None and target_iter is not None:
                try:
                    x_t, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    x_t, _ = next(target_iter)
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
                if temporal_radius > 0:
                    x_t = x_t[:, temporal_radius, ...]
                x_t = format_batch_for_model(x_t).to(device)
                logits_t = model(x_t)
                if isinstance(logits_t, (list, tuple)):
                    logits_t = torch.stack(logits_t, dim=0).mean(dim=0)
                if feature_extractor is not None:
                    feats_t = feature_extractor(x_t)['feat']
                    feats_t = feats_t.view(feats_t.size(0), -1)
                else:
                    feats_t = logits_t
                if args.mmd_classwise:
                    pseudo_t = logits_t.detach().argmax(dim=1)
                    mmd_cw = classwise_mmd_loss(center_feats, y, feats_t, pseudo_t, num_classes, mmd_sigmas)
                    mmd_global = mmd_loss(center_feats, feats_t, mmd_sigmas)
                    mmd_val = mmd_cw + mmd_global
                else:
                    mmd_val = mmd_loss(center_feats, feats_t, mmd_sigmas)
            if args.loss_mode == 'supcon':
                supcon = supervised_contrastive_loss(
                    center_feats,
                    y,
                    temperature=args.supcon_temp,
                    class_weight_tensor=class_weights,
                )
                ce = classification_loss(center_logits, y)
                base_loss = args.supcon_weight * supcon + args.ce_weight * ce
            else:
                base_loss = classification_loss(center_logits, y)
            if mmd_val is not None:
                base_loss = base_loss + args.mmd_weight * mmd_val
            temporal_loss = None
            if temporal_logits is not None:
                temporal_loss = classification_loss(temporal_logits, y)
            if not torch.isfinite(base_loss) or (temporal_loss is not None and not torch.isfinite(temporal_loss)):
                print('Warning: non-finite loss; skipping batch')
                continue
            if train:
                if not temporal_only:
                    base_loss.backward()
                    if args.grad_clip and args.grad_clip > 0:
                        clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    (optim_base or optimizer_base).step()
                if temporal_loss is not None and (optim_temp or optimizer_temporal) is not None:
                    (optim_temp or optimizer_temporal).zero_grad(set_to_none=True)
                    temporal_loss.backward()
                    (optim_temp or optimizer_temporal).step()
            total_loss += float(base_loss.item()) * y.size(0)
            total_base_loss += float(base_loss.item()) * y.size(0)
            if temporal_loss is not None:
                total_temporal_loss += float(temporal_loss.item()) * y.size(0)
            if mmd_val is not None:
                total_mmd_loss += float(mmd_val.item()) * y.size(0)
            preds = final_logits.argmax(dim=1)
            total_correct += int((preds == y).sum().item())
            total += y.size(0)
            all_preds.append(preds)
            all_targets.append(y)
            batch_idx += 1
            if args.dry_run or (args.limit_batches and batch_idx >= args.limit_batches):
                break
            loadbar.update(1)
        loadbar.close()

        avg_loss = total_loss / max(1, total)
        avg_base_loss = total_base_loss / max(1, total)
        avg_temporal_loss = total_temporal_loss / max(1, total) if temporal_head is not None else None
        acc = total_correct / max(1, total)
        avg_mmd_loss = total_mmd_loss / max(1, total) if mmd_active else None
        if len(all_preds) > 0:
            preds_cat = torch.cat(all_preds)
            targs_cat = torch.cat(all_targets)
            acc2, macro_f1, kappa, cm = compute_metrics(preds_cat, targs_cat, num_classes)
        else:
            preds_cat = torch.empty(0, dtype=torch.long)
            targs_cat = torch.empty(0, dtype=torch.long)
            macro_f1, kappa, cm = 0.0, 0.0, None
        details = None
        if collect_energy and energy_records and len(all_preds) > 0:
            energies = torch.cat(energy_records)
            details = {
                'preds': preds_cat.detach().cpu(),
                'targets': targs_cat.detach().cpu(),
                'energies': energies.detach().cpu(),
            }
        return avg_loss, avg_base_loss, avg_temporal_loss, avg_mmd_loss, acc, macro_f1, kappa, cm, details

    def load_checkpoint_for_eval(path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if 'model' not in checkpoint:
            raise KeyError('Checkpoint missing "model" key')
        model.load_state_dict(checkpoint['model'])
        if temporal_head is not None:
            if 'temporal_head' in checkpoint:
                temporal_head.load_state_dict(checkpoint['temporal_head'])
            else:
                alt_path = os.path.join(os.path.dirname(path), 'best_temporal.pt')
                if os.path.isfile(alt_path):
                    alt_ckpt = torch.load(alt_path, map_location=device, weights_only=False)
                    if 'temporal_head' in alt_ckpt:
                        temporal_head.load_state_dict(alt_ckpt['temporal_head'])
                    else:
                        print('Warning: temporal head checkpoint missing "temporal_head" key.')
                else:
                    print('Warning: checkpoint missing temporal head; temporal outputs may be random.')
        return checkpoint

    def evaluate_split(
        loader,
        split_name: str = 'eval',
        save_plot: bool = False,
        level_thresholds: Optional[Dict[float, float]] = None,
        level_reference_coverages: Optional[Dict[float, float]] = None,
    ):
        collect_energy = should_collect_energy()
        loss, base_loss, temporal_loss, mmd_loss_val, acc, macro_f1, kappa, cm, details = run_epoch(loader, train=False, collect_energy=collect_energy)
        if temporal_loss is not None:
            print(f"[{split_name}] base_loss {base_loss:.4f} temporal_loss {temporal_loss:.4f} acc {acc:.3f} f1 {macro_f1:.3f} kappa {kappa:.3f}")
        else:
            print(f"[{split_name}] loss {loss:.4f} acc {acc:.3f} f1 {macro_f1:.3f} kappa {kappa:.3f}")
        if save_plot and cm is not None:
            if label_mode == 'binary':
                class_names = ['Wake', 'Sleep']
            elif label_mode == 'n1':
                class_names = ['non-N1', 'N1']
            elif label_mode == 'sleep3':
                class_names = ['N2', 'N3', 'REM']
            elif label_mode == 'sleep4':
                class_names = ['N1', 'N2', 'N3', 'REM']
            elif num_classes == 5:
                class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
            else:
                class_names = [f'Class {i}' for i in range(num_classes)]
            cm_path = os.path.join(args.save_path, f'confusion_matrix_{split_name}.png')
            plot_and_save_cm(cm, class_names, cm_path, writer=None, tag=f'ConfusionMatrix/{split_name}', step=0)
        if collect_energy:
            log_uncertainty_metrics(
                split_name,
                details,
                level_thresholds=level_thresholds,
                level_reference_coverages=level_reference_coverages,
            )
        return loss, acc, macro_f1, kappa, cm

    def get_eval_loader_and_name():
        if test_loader is not None:
            return test_loader, 'test'
        if val_loader is not None:
            return val_loader, 'val'
        if train_loader is not None:
            return train_loader, 'train'
        eval_dataset = dataset
        if temporal_radius > 0:
            eval_dataset = TemporalContextDataset(dataset, indices=None, radius=temporal_radius)
        tmp_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(2, args.workers), pin_memory=True)
        return tmp_loader, 'full'

    def _class_names_for_mode():
        if label_mode == 'binary':
            return ['W', 'S']
        if label_mode == 'n1':
            return ['non-N1', 'N1']
        if label_mode == 'sleep3':
            return ['N2', 'N3', 'REM']
        if label_mode == 'sleep4':
            return ['N1', 'N2', 'N3', 'REM']
        return ['W', 'N1', 'N2', 'N3', 'REM']

    def _compute_energy_threshold(details, target_acc: float, default_thr: float) -> float:
        if details is None:
            return default_thr
        energies = details['energies'].detach().cpu().numpy()
        preds = details['preds'].detach().cpu().numpy()
        targets = details['targets'].detach().cpu().numpy()
        if energies.size == 0:
            return default_thr
        order = np.argsort(energies)
        energies_s = energies[order]
        correct = (preds == targets)[order]
        cum_correct = np.cumsum(correct)
        counts = np.arange(1, correct.size + 1)
        accs = cum_correct / counts
        ok = accs >= target_acc
        if not np.any(ok):
            return default_thr
        last_idx = np.where(ok)[0][-1]
        return float(energies_s[last_idx])

    def _energy_confidence_from_reference(energies_np: np.ndarray, reference_energies_np: np.ndarray) -> np.ndarray:
        if energies_np.size == 0:
            return np.asarray([], dtype=np.float32)
        if reference_energies_np.size == 0:
            return np.full_like(energies_np, 0.5, dtype=np.float32)
        ref_sorted = np.sort(reference_energies_np)
        ranks = np.searchsorted(ref_sorted, energies_np, side='right').astype(np.float32)
        coverage = ranks / max(float(ref_sorted.size), 1.0)
        conf = 1.0 - coverage
        return np.clip(conf, 0.0, 1.0).astype(np.float32)

    def _save_binary_routing_weights(details, base_indices, threshold: float, epoch: int, reference_details=None):
        if details is None or not base_indices:
            return
        energies = details['energies'].detach().cpu().numpy()
        preds = details['preds'].detach().cpu().numpy()
        ref_energies = energies
        if reference_details is not None and 'energies' in reference_details:
            ref_energies = reference_details['energies'].detach().cpu().numpy()
        conf_scores = _energy_confidence_from_reference(energies, ref_energies)
        n = int(min(len(base_indices), len(energies), len(preds)))
        if n <= 0:
            return
        if n < len(base_indices):
            print(f"Warning: routing export saw {n}/{len(base_indices)} train samples (likely due --limit_batches).")
        pred_sleep = (preds[:n] == 1).astype(np.float32)
        conf = conf_scores[:n]
        sleep_prob = conf * pred_sleep
        fallback_prob = 1.0 - conf
        sleep_w = 0.25 + 0.75 * sleep_prob
        fallback_w = 0.25 + 0.75 * fallback_prob
        weights = {}
        for i in range(n):
            base_idx = int(base_indices[i])
            key = dataset.sample_keys[base_idx] if hasattr(dataset, 'sample_keys') else str(base_idx)
            weights[key] = {
                'n1': float(sleep_w[i]),
                'sleep3': float(sleep_w[i]),
                'sleep4': float(sleep_w[i]),
                'fallback': float(fallback_w[i]),
                'confidence': float(conf[i]),
            }
        payload = {
            'version': 1,
            'stage': 'binary',
            'epoch': int(epoch),
            'threshold': float(threshold),
            'scheme': 'baseline_0.25_plus_0.75x_confidence',
            'weights': weights,
        }
        out_path = routing_weights_path or os.path.join(args.save_path, 'routing_weights.json')
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        print(f"Saved routing weights: {out_path} (keys={len(weights)})")

    def _load_checkpoint_model(path: str, num_classes: int, size: str) -> nn.Module:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')

        def _infer_convnext_size_from_checkpoint(ckpt_blob, fallback_size: str) -> str:
            valid_sizes = {'tiny', 'small', 'large'}
            if isinstance(ckpt_blob, dict):
                args_blob = ckpt_blob.get('args')
                if isinstance(args_blob, dict):
                    size_from_args = args_blob.get('convnext_size')
                    if isinstance(size_from_args, str) and size_from_args in valid_sizes:
                        return size_from_args
                state = ckpt_blob.get('model', ckpt_blob)
            else:
                state = ckpt_blob
            if not isinstance(state, dict):
                return fallback_size

            stem_w = state.get('features.0.0.weight')
            try:
                stem_out = int(stem_w.shape[0]) if stem_w is not None else -1
            except Exception:
                stem_out = -1
            if stem_out >= 192:
                return 'large'

            max_stage3_block = -1
            for key in state.keys():
                if not isinstance(key, str):
                    continue
                if key.startswith('features.5.'):
                    parts = key.split('.')
                    if len(parts) >= 3 and parts[2].isdigit():
                        max_stage3_block = max(max_stage3_block, int(parts[2]))
            if max_stage3_block >= 20:
                return 'small'
            if max_stage3_block >= 0:
                return 'tiny'
            return fallback_size

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        resolved_size = _infer_convnext_size_from_checkpoint(ckpt, size)
        if resolved_size != size:
            print(f"Checkpoint size override for {os.path.basename(path)}: cli={size} -> inferred={resolved_size}")

        model_local = build_convnext(num_classes=num_classes, size=resolved_size).to(device)
        adjust_convnext_input(model_local, num_modalities)
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        try:
            model_local.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(
                f"Failed loading checkpoint {path} with inferred size '{resolved_size}' (cli size '{size}'). "
                f"Check model-size and modality compatibility. Original error: {e}"
            ) from e
        model_local.eval()
        return model_local

    def _compute_metrics_np(preds_np: np.ndarray, targets_np: np.ndarray, out_num_classes: int):
        preds_t = torch.tensor(preds_np, dtype=torch.long)
        targets_t = torch.tensor(targets_np, dtype=torch.long)
        acc_v, f1_v, kappa_v, _ = compute_metrics(preds_t, targets_t, out_num_classes)
        return float(acc_v), float(f1_v), float(kappa_v)

    def _run_cascade_predictions(loader, bin_model: nn.Module, n1_model: nn.Module, sleep3_model: nn.Module, fallback_model: nn.Module):
        all_bin_pred = []
        all_n1_pred = []
        all_sleep3_pred = []
        all_fb_pred = []
        all_energy = []
        all_targets = []
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                if temporal_radius > 0:
                    x = x[:, temporal_radius, ...]
                x = format_batch_for_model(x).to(device)
                logits_bin = bin_model(x)
                if isinstance(logits_bin, (list, tuple)):
                    logits_bin = torch.stack(logits_bin, dim=0).mean(dim=0)
                logits_n1 = n1_model(x)
                if isinstance(logits_n1, (list, tuple)):
                    logits_n1 = torch.stack(logits_n1, dim=0).mean(dim=0)
                logits_sleep3 = sleep3_model(x)
                if isinstance(logits_sleep3, (list, tuple)):
                    logits_sleep3 = torch.stack(logits_sleep3, dim=0).mean(dim=0)
                logits_fb = fallback_model(x)
                if isinstance(logits_fb, (list, tuple)):
                    logits_fb = torch.stack(logits_fb, dim=0).mean(dim=0)
                all_bin_pred.append(logits_bin.argmax(dim=1).detach().cpu().numpy())
                all_n1_pred.append(logits_n1.argmax(dim=1).detach().cpu().numpy())
                all_sleep3_pred.append(logits_sleep3.argmax(dim=1).detach().cpu().numpy())
                all_fb_pred.append(logits_fb.argmax(dim=1).detach().cpu().numpy())
                all_energy.append(compute_energy(logits_bin).detach().cpu().numpy())
                all_targets.append(y.detach().cpu().numpy())
        if not all_targets:
            return None
        return {
            'bin_pred': np.concatenate(all_bin_pred, axis=0),
            'n1_pred': np.concatenate(all_n1_pred, axis=0),
            'sleep3_pred': np.concatenate(all_sleep3_pred, axis=0),
            'fb_pred': np.concatenate(all_fb_pred, axis=0),
            'energy': np.concatenate(all_energy, axis=0),
            'targets': np.concatenate(all_targets, axis=0),
        }

    def _compose_cascade_predictions(blob, threshold: float):
        bin_pred = blob['bin_pred']
        n1_pred = blob['n1_pred']
        sleep3_pred = blob['sleep3_pred']
        fb_pred = blob['fb_pred']
        energy = blob['energy']
        final_pred = np.array(fb_pred, copy=True)
        conf_mask = energy <= float(threshold)
        wake_mask = conf_mask & (bin_pred == 0)
        sleep_mask = conf_mask & (bin_pred == 1)
        n1_mask = sleep_mask & (n1_pred == 1)
        sleep3_mask = sleep_mask & (n1_pred == 0)
        final_pred[wake_mask] = 0
        final_pred[n1_mask] = 1
        if np.any(sleep3_mask):
            # sleep3 classes: N2->0, N3->1, REM->2
            map_arr = np.asarray([2, 3, 4], dtype=np.int64)
            final_pred[sleep3_mask] = map_arr[sleep3_pred[sleep3_mask]]
        return final_pred

    def _fallback_rate(blob, threshold: float) -> float:
        energies = blob['energy']
        if energies.size == 0:
            return 0.0
        conf_mask = energies <= float(threshold)
        return float(1.0 - np.mean(conf_mask.astype(np.float32)))

    def _score_tuple_for_metric(acc_v: float, f1_v: float, kappa_v: float):
        if args.cascade_metric == 'acc':
            return (acc_v, kappa_v, f1_v)
        if args.cascade_metric == 'f1':
            return (f1_v, kappa_v, acc_v)
        return (kappa_v, f1_v, acc_v)

    def _find_best_cascade_threshold(calib_blob, initial_threshold: float):
        energies = calib_blob['energy']
        targets = calib_blob['targets']
        if energies.size == 0:
            return float(initial_threshold), {'acc': 0.0, 'f1': 0.0, 'kappa': 0.0, 'n': 0}
        n_points = max(3, int(args.cascade_threshold_points))
        quantiles = np.linspace(0.0, 1.0, n_points)
        candidates = np.quantile(energies, quantiles).astype(np.float64)
        candidates = np.unique(np.concatenate([candidates, np.asarray([float(initial_threshold)], dtype=np.float64)]))
        best_threshold = float(initial_threshold)
        best_metrics = {'acc': 0.0, 'f1': 0.0, 'kappa': -1.0, 'fallback_rate': 1.0, 'n': int(targets.size)}
        best_score = None
        feasible_best_threshold = None
        feasible_best_metrics = None
        feasible_best_score = None
        max_fb_rate = float(np.clip(args.cascade_max_fallback_rate, 0.0, 1.0))
        for thr in candidates:
            preds_np = _compose_cascade_predictions(calib_blob, float(thr))
            acc_v, f1_v, kappa_v = _compute_metrics_np(preds_np, targets, 5)
            fb_rate = _fallback_rate(calib_blob, float(thr))
            score = _score_tuple_for_metric(acc_v, f1_v, kappa_v)
            metrics = {'acc': acc_v, 'f1': f1_v, 'kappa': kappa_v, 'fallback_rate': fb_rate, 'n': int(targets.size)}
            if fb_rate <= max_fb_rate:
                if feasible_best_score is None or score > feasible_best_score:
                    feasible_best_score = score
                    feasible_best_threshold = float(thr)
                    feasible_best_metrics = metrics
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = float(thr)
                best_metrics = metrics
        if feasible_best_threshold is not None and feasible_best_metrics is not None:
            return feasible_best_threshold, feasible_best_metrics
        return best_threshold, best_metrics

    def _load_cascade_threshold(path: str):
        if not path or not os.path.isfile(path):
            return None
        try:
            blob = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    blob = json.load(f)
            except Exception:
                return None
        if isinstance(blob, dict):
            if blob.get('binary_energy_threshold') is not None:
                return float(blob.get('binary_energy_threshold'))
            if blob.get('threshold') is not None:
                return float(blob.get('threshold'))
        return None

    if args.cascade_train:
        sleep3_checkpoint_path = args.sleep3_checkpoint if args.sleep3_checkpoint.strip() else args.sleep_checkpoint
        bin_model = _load_checkpoint_model(args.binary_checkpoint, num_classes=2, size=args.binary_convnext_size)
        n1_model = _load_checkpoint_model(args.n1_checkpoint, num_classes=2, size=args.n1_convnext_size)
        sleep3_model = _load_checkpoint_model(sleep3_checkpoint_path, num_classes=3, size=args.sleep3_convnext_size)
        fallback_model = _load_checkpoint_model(args.fallback_checkpoint, num_classes=5, size=args.fallback_convnext_size)

        if val_loader is not None:
            calib_loader = val_loader
            calib_name = 'val'
        elif train_loader is not None:
            calib_loader = train_loader
            calib_name = 'train'
        else:
            raise ValueError('Cascade training requires at least val or train split for calibration.')

        initial_thr = float(args.binary_energy_threshold_start)
        try:
            bckpt = torch.load(args.binary_checkpoint, map_location=device, weights_only=False)
            initial_thr = float(bckpt.get('binary_energy_threshold', initial_thr))
        except Exception:
            pass

        calib_blob = _run_cascade_predictions(calib_loader, bin_model, n1_model, sleep3_model, fallback_model)
        if calib_blob is None:
            raise RuntimeError('Cascade calibration split produced no predictions.')
        best_thr, best_metrics = _find_best_cascade_threshold(calib_blob, initial_thr)
        print(
            f"[cascade/train] split={calib_name} metric={args.cascade_metric} threshold={best_thr:.4f} "
            f"acc={best_metrics['acc']:.3f} f1={best_metrics['f1']:.3f} kappa={best_metrics['kappa']:.3f} "
            f"fallback={best_metrics.get('fallback_rate', float('nan')):.3f}"
        )

        if test_loader is not None:
            eval_loader = test_loader
            eval_name = 'test'
        elif val_loader is not None:
            eval_loader = val_loader
            eval_name = 'val'
        else:
            eval_loader = train_loader
            eval_name = 'train'

        eval_blob = _run_cascade_predictions(eval_loader, bin_model, n1_model, sleep3_model, fallback_model)
        eval_acc = eval_f1 = eval_kappa = float('nan')
        eval_fallback_rate = float('nan')
        if eval_blob is not None:
            eval_preds = _compose_cascade_predictions(eval_blob, best_thr)
            eval_acc, eval_f1, eval_kappa = _compute_metrics_np(eval_preds, eval_blob['targets'], 5)
            eval_fallback_rate = _fallback_rate(eval_blob, best_thr)
            print(f"[cascade/{eval_name}] acc {eval_acc:.3f} f1 {eval_f1:.3f} kappa {eval_kappa:.3f} fallback {eval_fallback_rate:.3f} (bin_thr {best_thr:.4f})")

        cascade_ckpt_path = args.cascade_checkpoint.strip() if args.cascade_checkpoint.strip() else os.path.join(args.save_path, 'cascade_best.pt')
        os.makedirs(os.path.dirname(cascade_ckpt_path) or '.', exist_ok=True)
        torch.save({
            'type': 'cascade_threshold',
            'epoch': 0,
            'binary_energy_threshold': float(best_thr),
            'calibration_split': calib_name,
            'calibration_metric': args.cascade_metric,
            'calibration_scores': best_metrics,
            'evaluation_split': eval_name,
            'evaluation_scores': {
                'acc': float(eval_acc),
                'f1': float(eval_f1),
                'kappa': float(eval_kappa),
                'fallback_rate': float(eval_fallback_rate),
            },
            'binary_checkpoint': args.binary_checkpoint,
            'n1_checkpoint': args.n1_checkpoint,
            'sleep3_checkpoint': sleep3_checkpoint_path,
            'fallback_checkpoint': args.fallback_checkpoint,
            'args': vars(args),
        }, cascade_ckpt_path)
        print(f"Saved cascade checkpoint: {cascade_ckpt_path}")
        _append_run_log({
            'event': 'cascade_train_end',
            'run_id': run_id,
            'timestamp_utc': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
            'cascade_checkpoint': cascade_ckpt_path,
            'binary_energy_threshold': float(best_thr),
            'calibration_split': calib_name,
            'calibration_metric': args.cascade_metric,
            'calibration_scores': best_metrics,
            'evaluation_split': eval_name,
            'evaluation_scores': {
                'acc': float(eval_acc),
                'f1': float(eval_f1),
                'kappa': float(eval_kappa),
                'fallback_rate': float(eval_fallback_rate),
            }
        })
        sys.exit(0)

    if args.test_only and args.cascade_eval:
        sleep3_checkpoint_path = args.sleep3_checkpoint if args.sleep3_checkpoint.strip() else args.sleep_checkpoint
        bin_model = _load_checkpoint_model(args.binary_checkpoint, num_classes=2, size=args.binary_convnext_size)
        n1_model = _load_checkpoint_model(args.n1_checkpoint, num_classes=2, size=args.n1_convnext_size)
        sleep3_model = _load_checkpoint_model(sleep3_checkpoint_path, num_classes=3, size=args.sleep3_convnext_size)
        fallback_model = _load_checkpoint_model(args.fallback_checkpoint, num_classes=5, size=args.fallback_convnext_size)
        eval_loader, split_name = get_eval_loader_and_name()
        bin_thr = args.binary_energy_threshold_start
        cascade_thr = _load_cascade_threshold(args.cascade_checkpoint.strip())
        if cascade_thr is not None:
            bin_thr = float(cascade_thr)
        try:
            ckpt = torch.load(args.binary_checkpoint, map_location=device, weights_only=False)
            if cascade_thr is None:
                bin_thr = float(ckpt.get('binary_energy_threshold', bin_thr))
        except Exception:
            pass
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in eval_loader:
                x, y = batch
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                if temporal_radius > 0:
                    x = x[:, temporal_radius, ...]
                x = format_batch_for_model(x).to(device)
                y = y.to(device)
                logits_bin = bin_model(x)
                if isinstance(logits_bin, (list, tuple)):
                    logits_bin = torch.stack(logits_bin, dim=0).mean(dim=0)
                energy = compute_energy(logits_bin)
                conf_mask = energy <= bin_thr
                bin_pred = logits_bin.argmax(dim=1)
                # fallback if binary is uncertain
                logits_fb = fallback_model(x)
                if isinstance(logits_fb, (list, tuple)):
                    logits_fb = torch.stack(logits_fb, dim=0).mean(dim=0)
                fb_pred = logits_fb.argmax(dim=1)
                # n1 and sleep3 only for confident sleep
                n1_pred = None
                sleep3_pred = None
                if torch.any(conf_mask & (bin_pred == 1)):
                    logits_n1 = n1_model(x)
                    if isinstance(logits_n1, (list, tuple)):
                        logits_n1 = torch.stack(logits_n1, dim=0).mean(dim=0)
                    n1_pred = logits_n1.argmax(dim=1)
                    logits_sleep3 = sleep3_model(x)
                    if isinstance(logits_sleep3, (list, tuple)):
                        logits_sleep3 = torch.stack(logits_sleep3, dim=0).mean(dim=0)
                    sleep3_pred = logits_sleep3.argmax(dim=1)
                final_pred = fb_pred.clone()
                # confident wake -> 0
                final_pred[conf_mask & (bin_pred == 0)] = 0
                if n1_pred is not None and sleep3_pred is not None:
                    sleep_mask = conf_mask & (bin_pred == 1)
                    n1_mask = sleep_mask & (n1_pred == 1)
                    sleep3_mask = sleep_mask & (n1_pred == 0)
                    final_pred[n1_mask] = 1
                    if torch.any(sleep3_mask):
                        sleep3_local = sleep3_pred[sleep3_mask]
                        mapped = torch.empty_like(sleep3_local)
                        mapped[sleep3_local == 0] = 2  # N2
                        mapped[sleep3_local == 1] = 3  # N3
                        mapped[sleep3_local == 2] = 4  # REM
                        final_pred[sleep3_mask] = mapped
                all_preds.append(final_pred.detach().cpu())
                all_targets.append(y.detach().cpu())
        preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
        targs_cat = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
        if preds_cat.numel() == 0:
            print('No predictions generated in cascade eval.')
            sys.exit(0)
        acc, macro_f1, kappa, cm = compute_metrics(preds_cat, targs_cat, 5)
        fallback_rate = float('nan')
        if all_targets:
            total_count = int(sum(t.numel() for t in all_targets))
            if total_count > 0:
                fallback_count = 0
                with torch.no_grad():
                    for batch in eval_loader:
                        x_tmp, _ = batch
                        x_tmp = torch.nan_to_num(x_tmp, nan=0.0, posinf=0.0, neginf=0.0)
                        if temporal_radius > 0:
                            x_tmp = x_tmp[:, temporal_radius, ...]
                        x_tmp = format_batch_for_model(x_tmp).to(device)
                        logits_tmp = bin_model(x_tmp)
                        if isinstance(logits_tmp, (list, tuple)):
                            logits_tmp = torch.stack(logits_tmp, dim=0).mean(dim=0)
                        fallback_count += int((compute_energy(logits_tmp) > bin_thr).sum().item())
                fallback_rate = float(fallback_count / total_count)
        print(f"[cascade/{split_name}] acc {acc:.3f} f1 {macro_f1:.3f} kappa {kappa:.3f} fallback {fallback_rate:.3f} (bin_thr {bin_thr:.4f})")
        class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        cm_path = os.path.join(args.save_path, f'confusion_matrix_cascade_{split_name}.png')
        plot_and_save_cm(cm, class_names, cm_path)
        sys.exit(0)

    if args.test_only:
        checkpoint_target = args.checkpoint_path or os.path.join(args.save_path, 'best.pt')
        checkpoint_blob = load_checkpoint_for_eval(checkpoint_target)

        calibrated_level_thresholds: Optional[Dict[float, float]] = None
        calibrated_level_coverages: Optional[Dict[float, float]] = None
        coverage_levels = list(energy_levels) if energy_levels else [100.0, 90.0, 80.0, 70.0, 60.0, 50.0]

        if args.track_energy and val_loader is not None and coverage_levels:
            _, _, _, _, _, _, _, _, val_details_for_cal = run_epoch(
                val_loader,
                train=False,
                collect_energy=True,
                target_loader=None,
                use_mmd=False
            )
            if val_details_for_cal is not None and 'energies' in val_details_for_cal:
                val_energies = val_details_for_cal['energies'].detach().cpu().numpy()
                calibrated_level_thresholds = _compute_energy_level_thresholds(val_energies, coverage_levels)
                calibrated_level_coverages = _compute_coverage_from_thresholds(val_energies, calibrated_level_thresholds)
                print(f"[val] calibrated energy thresholds for levels: {coverage_levels}")
                for lv in coverage_levels:
                    thr = calibrated_level_thresholds.get(float(lv))
                    if thr is not None:
                        ref_cov = calibrated_level_coverages.get(float(lv), float('nan')) if calibrated_level_coverages is not None else float('nan')
                        if np.isfinite(ref_cov):
                            print(f"[val] level {lv:.1f}% -> energy<= {thr:.4f} (actual val_cov={ref_cov:.3f})")
                        else:
                            print(f"[val] level {lv:.1f}% -> energy<= {thr:.4f}")
                try:
                    if isinstance(checkpoint_blob, dict):
                        checkpoint_blob['energy_level_thresholds'] = {str(k): float(v) for k, v in calibrated_level_thresholds.items()}
                        checkpoint_blob['energy_level_coverages_validation'] = {str(k): float(v) for k, v in (calibrated_level_coverages or {}).items()}
                        checkpoint_blob['energy_level_source'] = 'validation'
                        checkpoint_blob['energy_level_levels'] = [float(x) for x in coverage_levels]
                        torch.save(checkpoint_blob, checkpoint_target)
                        print(f"Saved calibrated energy thresholds into checkpoint: {checkpoint_target}")
                except Exception as e:
                    print(f"Warning: could not persist calibrated thresholds into checkpoint: {e}")
        elif args.track_energy and isinstance(checkpoint_blob, dict):
            stored = checkpoint_blob.get('energy_level_thresholds')
            if isinstance(stored, dict):
                tmp: Dict[float, float] = {}
                for k, v in stored.items():
                    try:
                        tmp[float(k)] = float(v)
                    except Exception:
                        continue
                if tmp:
                    calibrated_level_thresholds = tmp
                    print(f"Loaded calibrated energy thresholds from checkpoint: {checkpoint_target}")
            stored_cov = checkpoint_blob.get('energy_level_coverages_validation')
            if isinstance(stored_cov, dict):
                tmp_cov: Dict[float, float] = {}
                for k, v in stored_cov.items():
                    try:
                        tmp_cov[float(k)] = float(v)
                    except Exception:
                        continue
                if tmp_cov:
                    calibrated_level_coverages = tmp_cov

        eval_loader, split_name = get_eval_loader_and_name()
        active_thresholds = calibrated_level_thresholds if split_name == 'test' else None
        active_ref_coverages = calibrated_level_coverages if split_name == 'test' else None
        loss, acc, macro_f1, kappa, _ = evaluate_split(
            eval_loader,
            split_name,
            save_plot=True,
            level_thresholds=active_thresholds,
            level_reference_coverages=active_ref_coverages,
        )
        if args.visualize_model:
            visualize_tsne(eval_loader, split_name)
        _append_run_log({
            'event': 'run_end',
            'run_id': run_id,
            'timestamp_utc': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
            'mode': 'test_only',
            'split': split_name,
            'loss': loss,
            'acc': acc,
            'f1': macro_f1,
            'kappa': kappa,
            'checkpoint': checkpoint_target
        })
        sys.exit(0)

    if args.dry_run:
        print('Running dry-run...')
        x0, y0 = next(iter(train_loader))
        x0 = torch.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
        if temporal_radius > 0:
            B0, K0, M0, F0, T0 = x0.shape
            x0_full = x0
            x0 = x0_full[:, temporal_radius, ...]
        x0 = format_batch_for_model(x0).to(device)
        y0 = y0.to(device)
        logits = model(x0)
        if isinstance(logits, (list, tuple)):
            logits = torch.stack(logits, dim=0).mean(dim=0)
        if temporal_radius > 0:
            center_logits = logits
            temporal_logits = None
            if temporal_head is not None:
                x0_full = x0_full.view(B0 * K0, M0, F0, T0)
                x0_full = format_batch_for_model(x0_full).to(device)
                with torch.no_grad():
                    if feature_extractor is not None:
                        feats_full = feature_extractor(x0_full)['feat']
                        feats_full = feats_full.view(feats_full.size(0), -1)
                        feats_full = feats_full.view(y0.size(0), 2 * temporal_radius + 1, -1)
                    else:
                        logits_full = model(x0_full)
                        if isinstance(logits_full, (list, tuple)):
                            logits_full = torch.stack(logits_full, dim=0).mean(dim=0)
                        logits_full = logits_full.view(y0.size(0), 2 * temporal_radius + 1, -1)
                        feats_full = logits_full
                temporal_logits = temporal_head(feats_full.reshape(y0.size(0), -1))
            optimizer_base.zero_grad(set_to_none=True)
            base_loss = classification_loss(center_logits, y0)
            base_loss.backward()
            optimizer_base.step()
            if temporal_logits is not None and optimizer_temporal is not None:
                optimizer_temporal.zero_grad(set_to_none=True)
                temporal_loss = classification_loss(temporal_logits, y0)
                temporal_loss.backward()
                optimizer_temporal.step()
                print('Dry-run OK: logits', tuple(logits.shape), 'base_loss', float(base_loss.item()), 'temporal_loss', float(temporal_loss.item()))
            else:
                print('Dry-run OK: logits', tuple(logits.shape), 'base_loss', float(base_loss.item()))
        else:
            optimizer_base.zero_grad(set_to_none=True)
            base_loss = classification_loss(logits, y0)
            base_loss.backward()
            optimizer_base.step()
            print('Dry-run OK: logits', tuple(logits.shape), 'loss', float(base_loss.item()))
    else:
        print('========= Training =========')
        writer = None
        if TB_AVAILABLE:
            os.makedirs(args.log_dir, exist_ok=True)
            try:
                writer = SummaryWriter(log_dir=args.log_dir)  # type: ignore[name-defined]
            except Exception:
                writer = None
        else:
            print("TensorBoard not available")
        best_va = None
        best_path = os.path.join(args.save_path, 'best.pt')
        last_path = os.path.join(args.save_path, 'last.pt')
        best_base_path = os.path.join(args.save_path, 'best_base.pt')
        best_temporal_path = os.path.join(args.save_path, 'best_temporal.pt')
        last_base_path = os.path.join(args.save_path, 'last_base.pt')
        last_temporal_path = os.path.join(args.save_path, 'last_temporal.pt')
        loadbar = tqdm(total=args.epochs, desc="Epochs", position=0)
        collect_energy = should_collect_energy()
        damnet_enabled = args.domain_adapt == 'damnet' and target_micro_loader is not None and target_val_loader is not None
        micro_lr = args.micro_lr if args.micro_lr and args.micro_lr > 0 else args.lr * 0.1
        micro_optimizer_base = torch.optim.AdamW(list(model.parameters()), lr=micro_lr, weight_decay=1e-2)
        micro_optimizer_temporal = None
        if temporal_head is not None:
            micro_optimizer_temporal = torch.optim.AdamW(list(temporal_head.parameters()), lr=micro_lr, weight_decay=1e-2)
        micro_scheduler_base = build_lr_scheduler(micro_optimizer_base, args.epochs)
        micro_scheduler_temporal = build_lr_scheduler(micro_optimizer_temporal, args.epochs)
        binary_energy_threshold = args.binary_energy_threshold_start
        for epoch in range(1, args.epochs + 1):
            current_lr_base = float((optimizer_base.param_groups[0]).get('lr', args.lr))
            current_lr_temporal = float((optimizer_temporal.param_groups[0]).get('lr', args.lr)) if optimizer_temporal is not None else None
            tr_loss, tr_base_loss, tr_temporal_loss, tr_mmd_loss, tr_acc, tr_f1, tr_kappa, _, tr_details = run_epoch(
                train_loader,
                train=True,
                collect_energy=bool(args.track_energy or label_mode == 'binary'),
                target_loader=target_loader
            )
            va_loss, va_base_loss, va_temporal_loss, va_mmd_loss, va_acc, va_f1, va_kappa, va_cm, va_details = run_epoch(
                val_loader,
                train=False,
                collect_energy=collect_energy
            )
            if damnet_enabled:
                micro_loss, micro_base_loss, micro_temporal_loss, _, micro_acc, micro_f1, micro_kappa, _, _ = run_epoch(
                    target_micro_loader,
                    train=True,
                    collect_energy=False,
                    target_loader=None,
                    optim_base=micro_optimizer_base,
                    optim_temp=micro_optimizer_temporal,
                    use_mmd=False
                )
                tgt_loss, tgt_base_loss, tgt_temporal_loss, _, tgt_acc, tgt_f1, tgt_kappa, tgt_cm, tgt_details = run_epoch(
                    target_val_loader,
                    train=False,
                    collect_energy=collect_energy,
                    target_loader=None,
                    use_mmd=False
                )
            _append_run_log({
                'event': 'generation_stats',
                'run_id': run_id,
                'timestamp_utc': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
                'generation': epoch,
                'fitness': va_loss,
                'val_loss': va_loss,
                'val_base_loss': va_base_loss,
                'val_temporal_loss': va_temporal_loss,
                'val_mmd_loss': va_mmd_loss,
                'val_acc': va_acc,
                'val_f1': va_f1,
                'val_kappa': va_kappa,
                'train_loss': tr_loss,
                'train_base_loss': tr_base_loss,
                'train_temporal_loss': tr_temporal_loss,
                'train_mmd_loss': tr_mmd_loss,
                'lr_base': current_lr_base,
                'lr_temporal': current_lr_temporal,
                'train_acc': tr_acc,
                'train_f1': tr_f1,
                'train_kappa': tr_kappa,
            })
            if damnet_enabled:
                _append_run_log({
                    'event': 'damnet_target_stats',
                    'run_id': run_id,
                    'timestamp_utc': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
                    'generation': epoch,
                    'micro_loss': micro_loss,
                    'micro_base_loss': micro_base_loss,
                    'micro_temporal_loss': micro_temporal_loss,
                    'micro_acc': micro_acc,
                    'micro_f1': micro_f1,
                    'micro_kappa': micro_kappa,
                    'target_val_loss': tgt_loss,
                    'target_val_base_loss': tgt_base_loss,
                    'target_val_temporal_loss': tgt_temporal_loss,
                    'target_val_acc': tgt_acc,
                    'target_val_f1': tgt_f1,
                    'target_val_kappa': tgt_kappa,
                })
            if tr_temporal_loss is not None and va_temporal_loss is not None:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"lr {current_lr_base:.2e}" +
                    (f"/{current_lr_temporal:.2e}" if current_lr_temporal is not None else "") +
                    " | " +
                    f"train base {tr_base_loss:.4f} temporal {tr_temporal_loss:.4f}" +
                    (f" mmd {tr_mmd_loss:.4f}" if tr_mmd_loss is not None else "") +
                    f" acc {tr_acc:.3f} f1 {tr_f1:.3f} kappa {tr_kappa:.3f} | "
                    f"val base {va_base_loss:.4f} temporal {va_temporal_loss:.4f}" +
                    (f" mmd {va_mmd_loss:.4f}" if va_mmd_loss is not None else "") +
                    f" acc {va_acc:.3f} f1 {va_f1:.3f} kappa {va_kappa:.3f}"
                )
            else:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"lr {current_lr_base:.2e}" +
                    (f"/{current_lr_temporal:.2e}" if current_lr_temporal is not None else "") +
                    " | " +
                    f"train loss {tr_loss:.4f}" +
                    (f" mmd {tr_mmd_loss:.4f}" if tr_mmd_loss is not None else "") +
                    f" acc {tr_acc:.3f} f1 {tr_f1:.3f} kappa {tr_kappa:.3f} | "
                    f"val loss {va_loss:.4f}" +
                    (f" mmd {va_mmd_loss:.4f}" if va_mmd_loss is not None else "") +
                    f" acc {va_acc:.3f} f1 {va_f1:.3f} kappa {va_kappa:.3f}"
                )
            if damnet_enabled:
                print(
                    f"DAM-Net target | micro loss {micro_loss:.4f} acc {micro_acc:.3f} f1 {micro_f1:.3f} kappa {micro_kappa:.3f} | "
                    f"target val loss {tgt_loss:.4f} acc {tgt_acc:.3f} f1 {tgt_f1:.3f} kappa {tgt_kappa:.3f}"
                )
            if args.track_energy:
                log_uncertainty_metrics('train', tr_details, epoch)
            if collect_energy:
                log_uncertainty_metrics('val', va_details, epoch)
                if label_mode == 'binary':
                    if epoch <= max(1, int(args.binary_energy_warmup_epochs)):
                        binary_energy_threshold = float(args.binary_energy_threshold_start)
                    else:
                        threshold_source = va_details if va_details is not None else tr_details
                        binary_energy_threshold = _compute_energy_threshold(
                            threshold_source,
                            target_acc=float(args.binary_energy_target_acc),
                            default_thr=binary_energy_threshold
                        )
                    print(f"Epoch {epoch}: binary energy threshold={binary_energy_threshold:.4f}")
                if args.track_energy:
                    plot_energy_accuracy_curve(va_details, 'val', epoch)
                if damnet_enabled:
                    log_uncertainty_metrics('target_val', tgt_details, epoch)
                    if args.track_energy:
                        plot_energy_accuracy_curve(tgt_details, 'target_val', epoch)
            # TensorBoard logs
            if writer is not None:
                writer.add_scalar('Loss/train', tr_loss, epoch)
                writer.add_scalar('Loss/train_base', tr_base_loss, epoch)
                if tr_temporal_loss is not None:
                    writer.add_scalar('Loss/train_temporal', tr_temporal_loss, epoch)
                if tr_mmd_loss is not None:
                    writer.add_scalar('Loss/train_mmd', tr_mmd_loss, epoch)
                writer.add_scalar('LR/base', current_lr_base, epoch)
                if current_lr_temporal is not None:
                    writer.add_scalar('LR/temporal', current_lr_temporal, epoch)
                writer.add_scalar('Acc/train', tr_acc, epoch)
                writer.add_scalar('F1/train', tr_f1, epoch)
                writer.add_scalar('Kappa/train', tr_kappa, epoch)
                writer.add_scalar('Loss/val', va_loss, epoch)
                writer.add_scalar('Loss/val_base', va_base_loss, epoch)
                if va_temporal_loss is not None:
                    writer.add_scalar('Loss/val_temporal', va_temporal_loss, epoch)
                if va_mmd_loss is not None:
                    writer.add_scalar('Loss/val_mmd', va_mmd_loss, epoch)
                writer.add_scalar('Acc/val', va_acc, epoch)
                writer.add_scalar('F1/val', va_f1, epoch)
                writer.add_scalar('Kappa/val', va_kappa, epoch)
            # Save confusion matrix plot for val
            if va_cm is not None:
                class_names = _class_names_for_mode()
                
                cm_path = os.path.join(args.save_path, f'confusion_matrix_epoch_{epoch}.png')
                # only plot on large steps to reduce overhead
                if epoch % 2 == 0:
                    plot_and_save_cm(va_cm, class_names, cm_path, writer=writer, tag='ConfusionMatrix/val', step=epoch)
            if damnet_enabled and tgt_cm is not None:
                class_names = _class_names_for_mode()
                cm_path = os.path.join(args.save_path, f'confusion_matrix_target_epoch_{epoch}.png')
                if epoch % 2 == 0:
                    plot_and_save_cm(tgt_cm, class_names, cm_path, writer=writer, tag='ConfusionMatrix/target', step=epoch)
            score = (va_kappa, va_f1, va_acc)
            if best_va is None or score > best_va:
                best_va = score
                torch.save({
                    'model': model.state_dict(),
                    'temporal_head': temporal_head.state_dict() if temporal_head is not None else None,
                    'epoch': epoch,
                    'score': score,
                    'binary_energy_threshold': binary_energy_threshold if label_mode == 'binary' else None,
                    'args': vars(args)
                }, best_path)
                if not temporal_only:
                    torch.save({'model': model.state_dict()}, best_base_path)
                if temporal_head is not None:
                    torch.save({'temporal_head': temporal_head.state_dict()}, best_temporal_path)
            torch.save({
                'model': model.state_dict(),
                'temporal_head': temporal_head.state_dict() if temporal_head is not None else None,
                'epoch': epoch,
                'score': score,
                'binary_energy_threshold': binary_energy_threshold if label_mode == 'binary' else None,
                'args': vars(args)
            }, last_path)
            if not temporal_only:
                torch.save({'model': model.state_dict()}, last_base_path)
            if temporal_head is not None:
                torch.save({'temporal_head': temporal_head.state_dict()}, last_temporal_path)

            if scheduler_base is not None:
                scheduler_base.step()
            if scheduler_temporal is not None:
                scheduler_temporal.step()
            if micro_scheduler_base is not None:
                micro_scheduler_base.step()
            if micro_scheduler_temporal is not None:
                micro_scheduler_temporal.step()
            loadbar.update(1)
        loadbar.close()
        if writer is not None:
            writer.close()

        if label_mode == 'binary' and args.save_routing_weights and train_eval_loader is not None:
            try:
                ckpt = load_checkpoint_for_eval(best_path)
                best_thr = float(ckpt.get('binary_energy_threshold', binary_energy_threshold))
            except Exception as e:
                print('Warning: could not load best checkpoint for routing-weight export:', e)
                best_thr = float(binary_energy_threshold)
            val_ref_details = None
            if val_loader is not None:
                _, _, _, _, _, _, _, _, val_ref_details = run_epoch(
                    val_loader,
                    train=False,
                    collect_energy=True,
                    target_loader=None,
                    use_mmd=False
                )
            _, _, _, _, _, _, _, _, train_eval_details = run_epoch(
                train_eval_loader,
                train=False,
                collect_energy=True,
                target_loader=None,
                use_mmd=False
            )
            _save_binary_routing_weights(
                train_eval_details,
                train_eval_base_indices,
                threshold=best_thr,
                epoch=int(ckpt.get('epoch', args.epochs)) if 'ckpt' in locals() and isinstance(ckpt, dict) else args.epochs,
                reference_details=val_ref_details
            )

        if test_loader is not None and len(test_loader.dataset) > 0:
            try:
                load_checkpoint_for_eval(best_path)
            except Exception as e:
                print('Warning: could not load best checkpoint for test evaluation:', e)
            evaluate_split(test_loader, 'test', save_plot=True)
        print('Training finished.')
        if args.visualize_model:
            visualize_tsne(val_loader, 'val', epoch=args.epochs)
        _append_run_log({
            'event': 'run_end',
            'run_id': run_id,
            'timestamp_utc': datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
            'mode': 'train',
            'best_score': best_va,
            'best_path': best_path,
            'best_base_path': os.path.join(args.save_path, 'best_base.pt'),
            'best_temporal_path': os.path.join(args.save_path, 'best_temporal.pt')
        })
