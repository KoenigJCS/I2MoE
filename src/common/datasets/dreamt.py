import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import (
	Compose,
	GaussianBlur,
	Normalize,
	RandomCrop,
	RandomHorizontalFlip,
	RandomVerticalFlip,
	Resize,
	ToTensor,
)
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from src.common.modules.common import Linear, PatchEmbeddings


TOKEN_TO_MODALITY = {
	# Wearable E4 channels
	"B": ("bvp", ["BVP", "bvp"]),
	"X": ("acc_x", ["ACC_X", "acc_x"]),
	"Y": ("acc_y", ["ACC_Y", "acc_y"]),
	"Z": ("acc_z", ["ACC_Z", "acc_z"]),
	"D": ("eda", ["EDA", "eda"]),
	"T": ("temp", ["TEMP", "temp"]),
	"H": ("hr", ["HR", "hr"]),
	"I": ("ibi", ["IBI", "ibi"]),

	# PSG channels
	"J": ("eeg_c4_m1", ["C4-M1", "C4_M1", "c4-m1", "c4_m1"]),
	"K": ("eeg_f4_m1", ["F4-M1", "F4_M1", "f4-m1", "f4_m1"]),
	"L": ("eeg_o2_m1", ["O2-M1", "O2_M1", "o2-m1", "o2_m1"]),
	"M": ("eeg_t3_cz", ["T3-CZ", "T3_CZ", "t3-cz", "t3_cz"]),
	"N": ("eeg_cz_t4", ["CZ-T4", "CZ_T4", "cz-t4", "cz_t4"]),
	"P": ("eog_e1", ["E1", "e1"]),
	"Q": ("eog_e2", ["E2", "e2"]),
	"C": ("chin", ["CHIN", "chin", "EMG", "emg", "EMG_submental"]),
	"E": ("ecg", ["ECG", "ecg"]),
	"R": ("ptaf", ["PTAF", "ptaf"]),
	"F": ("flow", ["FLOW", "flow", "Resp", "RESP", "resp"]),
	"U": ("thorax", ["THORAX", "thorax"]),
	"V": ("abdomen", ["ABDOMEN", "abdomen"]),
	"W": ("snore", ["SNORE", "snore"]),
	"G": ("lat", ["LAT", "lat"]),
	"A": ("rat", ["RAT", "rat"]),
	"O": ("sao2", ["SAO2", "sao2", "SpO2", "SPO2", "spo2"]),
}


def _parse_modality_groups(modality_arg):
	raw = str(modality_arg).strip()
	if not raw:
		return []

	if "," in raw:
		groups = [g.strip().upper() for g in raw.split(",") if g.strip()]
	else:
		groups = [ch.upper() for ch in raw if ch.strip()]

	parsed = []
	for group in groups:
		tokens = [ch for ch in group if ch in TOKEN_TO_MODALITY]
		if len(tokens) == 0:
			continue
		parsed.append(tokens)
	return parsed


def _group_name(tokens):
	joined = "_".join([TOKEN_TO_MODALITY[t][0] for t in tokens])
	return f"group_{joined}"


def _stack_group_channels(raw_modalities, tokens, sample_idx):
	parts = [raw_modalities[t][sample_idx] for t in tokens]
	return np.concatenate(parts, axis=2)


def _array_to_resized_tensor(arr, img_dim_y, img_dim_x, normalize_image):
	tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
	tensor = Fnn.interpolate(
		tensor.unsqueeze(0),
		size=(img_dim_y, img_dim_x),
		mode="bilinear",
		align_corners=False,
	).squeeze(0)
	if normalize_image:
		mean = torch.full((tensor.shape[0], 1, 1), 0.5, dtype=tensor.dtype)
		std = torch.full((tensor.shape[0], 1, 1), 0.5, dtype=tensor.dtype)
		tensor = (tensor - mean) / std
	return tensor


def _build_transforms(img_dim_x, img_dim_y, normalize_image):
	train_ops = [
		ToTensor(),
		Resize((img_dim_y, img_dim_x)),
	]
	eval_ops = [ToTensor(), Resize((img_dim_y, img_dim_x))]
	if normalize_image:
		norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		train_ops.append(norm)
		eval_ops.append(norm)
	return Compose(train_ops), Compose(eval_ops)


def _canonical_label(raw_label):
	label = str(raw_label).strip().upper()
	if label in {"R", "REM"}:
		return "REM"
	if label in {"W", "N1", "N2", "N3", "REM"}:
		return label
	return None


def _label_to_idx(raw_label):
	canon = _canonical_label(raw_label)
	label_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
	return label_map.get(canon, -1)


def _spec_target_shape(sr, segment_seconds, n_fft, hop_length):
	segment_samples = sr * segment_seconds
	pad = n_fft // 2
	freq_bins = n_fft // 2 + 1
	time_frames = 1 + max(
		0, (segment_samples + 2 * pad - n_fft) // max(1, hop_length)
	)
	return freq_bins, time_frames


def _vector_to_rgb_spec(vector, freq_bins, time_frames):
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


def _read_preprocessed_pairs(data_dir):
	pre_csvs = sorted(
		[
			p
			for p in data_dir.glob("preprocessed_*.csv")
			if not p.name.startswith("preprocessed_non_signal_")
		]
	)
	pairs = []
	for p in pre_csvs:
		base = p.name.replace("preprocessed_", "", 1)
		non_signal = data_dir / f"preprocessed_non_signal_{base}"
		if non_signal.exists():
			pairs.append((p, non_signal, base))
	return pairs


def _resolve_modality_columns(signal_df, modality_tokens):
	resolved = {}
	for token in modality_tokens:
		key = token.upper()
		canonical, candidates = TOKEN_TO_MODALITY.get(
			key, (key.lower(), [key, key.lower()])
		)
		found = next((c for c in candidates if c in signal_df.columns), None)
		if found is None:
			return None, None
		resolved[key] = (canonical, found)
	return resolved, None


def _extract_segment_labels(non_signal_df, sr, segment_seconds):
	if "Sleep_Stage" not in non_signal_df.columns:
		return None
	labels = non_signal_df["Sleep_Stage"].to_numpy()
	labels = np.array([_label_to_idx(v) for v in labels], dtype=np.int64)
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
	return np.array(seg_labels, dtype=np.int64)


def _build_dreamt_samples(
	data_dir,
	modality_tokens,
	max_files,
	sr,
	segment_seconds,
	n_fft,
	hop_length,
):
	pairs = _read_preprocessed_pairs(data_dir)
	if max_files > 0:
		pairs = pairs[:max_files]
	freq_bins, time_frames = _spec_target_shape(sr, segment_seconds, n_fft, hop_length)

	raw_modalities = {token: [] for token in modality_tokens}
	labels = []

	pair_iter = tqdm(pairs, desc="DREAMT: reading preprocessed files", leave=False)
	for sig_path, non_sig_path, _ in pair_iter:
		signal_df = pd.read_csv(sig_path)
		non_signal_df = pd.read_csv(non_sig_path)
		seg_labels = _extract_segment_labels(non_signal_df, sr, segment_seconds)
		if seg_labels is None:
			continue

		resolved, _ = _resolve_modality_columns(signal_df, modality_tokens)
		if resolved is None:
			continue

		n_segments = seg_labels.shape[0]
		per_token_specs = {}
		can_use_file = True
		for token in modality_tokens:
			_, col = resolved[token]
			stream = signal_df[col].to_numpy(dtype=np.float32).reshape(-1)
			stream = np.nan_to_num(stream, nan=0.0, posinf=0.0, neginf=0.0)
			chunk_len = len(stream) // max(1, n_segments)
			if chunk_len <= 0:
				can_use_file = False
				break
			specs = []
			for idx in range(n_segments):
				start = idx * chunk_len
				end = (idx + 1) * chunk_len
				specs.append(
					_vector_to_rgb_spec(stream[start:end], freq_bins, time_frames)
				)
			per_token_specs[token] = specs

		if not can_use_file:
			continue

		keep_mask = seg_labels >= 0
		if not np.any(keep_mask):
			continue

		kept_labels = seg_labels[keep_mask]
		for token in modality_tokens:
			token_specs = np.array(per_token_specs[token], dtype=np.uint8)[keep_mask]
			raw_modalities[token].extend(list(token_specs))
		labels.extend(list(kept_labels))

	if len(labels) == 0:
		raise ValueError(
			f"No DREAMT samples found in {data_dir}. "
			"Expected preprocessed_*.csv and matching preprocessed_non_signal_*.csv files."
		)

	return raw_modalities, np.array(labels, dtype=np.int64)


def load_and_preprocess_data_dreamt(args):
	data_dir = Path(getattr(args, "dreamt_data_dir", "data/dreamt"))
	if not data_dir.exists():
		raise FileNotFoundError(
			f"DREAMT data directory not found: {data_dir}. "
			"Set args.dreamt_data_dir to your preprocessed DREAMT root."
		)

	# Fixed DREAMT preprocessing parameters requested by user.
	sr = int(getattr(args, "dreamt_sr", 100))
	n_fft = int(getattr(args, "dreamt_n_fft", 256))
	hop_length = int(getattr(args, "dreamt_hop_length", 64))
	win_length = int(getattr(args, "dreamt_win_length", 256))
	if win_length != n_fft:
		win_length = n_fft

	segment_seconds = int(getattr(args, "dreamt_segment_seconds", 30))
	max_files = int(getattr(args, "dreamt_max_files", 0))

	img_dim_x = int(getattr(args, "dreamt_img_dim_x", 128))
	img_dim_y = int(getattr(args, "dreamt_img_dim_y", 256))
	random_seed = int(getattr(args, "dreamt_random_seed", 42))
	train_split = float(getattr(args, "dreamt_train_split", 0.65))
	val_split = float(getattr(args, "dreamt_val_split", 0.15))
	normalize_image = bool(getattr(args, "dreamt_normalize_image", False))

	modality_groups = _parse_modality_groups(args.modality)
	if len(modality_groups) == 0:
		raise ValueError("args.modality must include at least one modality token.")

	flat_tokens = sorted(set([t for grp in modality_groups for t in grp]))

	raw_modalities, labels = _build_dreamt_samples(
		data_dir,
		flat_tokens,
		max_files,
		sr,
		segment_seconds,
		n_fft,
		hop_length,
	)

	tqdm.write(
		f"DREAMT loader: discovered {labels.shape[0]} labeled segments "
		f"from {data_dir} using modalities {''.join(flat_tokens)}"
	)
	selected_names = [
		TOKEN_TO_MODALITY.get(t, (t.lower(), []))[0] for t in flat_tokens
	]
	tqdm.write(f"DREAMT loader: selected channels = {selected_names}")
	group_descriptions = ["".join(g) for g in modality_groups]
	tqdm.write(f"DREAMT loader: modality groups (experts) = {group_descriptions}")

	n_samples = labels.shape[0]
	all_idxs = list(range(n_samples))
	random.Random(random_seed).shuffle(all_idxs)
	train_stop = int(n_samples * train_split)
	val_stop = int(n_samples * (train_split + val_split))
	split_to_idxs = {
		"train": all_idxs[:train_stop],
		"val": all_idxs[train_stop:val_stop],
		"test": all_idxs[val_stop:],
	}

	train_set = set(split_to_idxs["train"])
	img_transforms_train, img_transforms_val_test = _build_transforms(
		img_dim_x, img_dim_y, normalize_image
	)

	data_dict = {}
	encoder_dict = {}
	input_dims = {}
	transforms = {}
	masks = {}

	observed_idx_arr = np.zeros((n_samples, len(modality_groups)), dtype=bool)
	modality_combinations = ["" for _ in range(n_samples)]

	for group_idx, group_tokens in enumerate(modality_groups):
		modality_name = _group_name(group_tokens)
		modality_data = []
		tqdm.write(f"DREAMT loader: building tensors for modality '{modality_name}'")
		sample_iter = tqdm(
			range(n_samples),
			desc=f"DREAMT: {modality_name} tensors",
			leave=False,
		)
		for idx in sample_iter:
			group_arr = _stack_group_channels(raw_modalities, group_tokens, idx)
			tensor = _array_to_resized_tensor(
				group_arr,
				img_dim_y,
				img_dim_x,
				normalize_image,
			)
			modality_data.append(tensor.numpy())
			observed_idx_arr[idx, group_idx] = True
			modality_combinations[idx] += modality_name
		data_dict[modality_name] = np.stack(modality_data).astype(np.float32)

	device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
	use_pretrained_convnext = bool(getattr(args, "dreamt_convnext_pretrained", False))
	freeze_convnext_features = bool(getattr(args, "dreamt_freeze_features", True))
	convnext_feature_dim = int(getattr(args, "dreamt_convnext_feature_dim", 1024))

	def _build_convnext_encoder_backbone(in_channels):
		weights = ConvNeXt_Tiny_Weights.DEFAULT if use_pretrained_convnext else None
		backbone = convnext_tiny(weights=weights)
		if in_channels != 3:
			old_conv = backbone.features[0][0]
			new_conv = torch.nn.Conv2d(
				in_channels,
				old_conv.out_channels,
				kernel_size=old_conv.kernel_size,
				stride=old_conv.stride,
				padding=old_conv.padding,
				bias=(old_conv.bias is not None),
			)
			with torch.no_grad():
				if use_pretrained_convnext:
					repeat_count = (in_channels + 2) // 3
					w = old_conv.weight.repeat(1, repeat_count, 1, 1)[:, :in_channels]
					w = w * (3.0 / float(in_channels))
					new_conv.weight.copy_(w)
					if old_conv.bias is not None:
						new_conv.bias.copy_(old_conv.bias)
				else:
					torch.nn.init.kaiming_normal_(
						new_conv.weight, mode="fan_out", nonlinearity="relu"
					)
					if new_conv.bias is not None:
						new_conv.bias.zero_()
			backbone.features[0][0] = new_conv
		in_features = backbone.classifier[2].in_features
		backbone.classifier[2] = torch.nn.Linear(in_features, convnext_feature_dim)
		if freeze_convnext_features:
			for p in backbone.features.parameters():
				p.requires_grad = False
		return backbone.to(device)

	for group_tokens in modality_groups:
		modality_name = _group_name(group_tokens)
		convnext_backbone = _build_convnext_encoder_backbone(3 * len(group_tokens))
		if args.patch:
			encoder_dict[modality_name] = torch.nn.Sequential(
				convnext_backbone,
				PatchEmbeddings(
					convnext_feature_dim,
					num_patches=args.num_patches,
					embed_dim=args.hidden_dim,
				).to(device),
			)
		else:
			encoder_dict[modality_name] = torch.nn.Sequential(
				convnext_backbone,
				Linear(convnext_feature_dim, args.hidden_dim, xavier_init=True).to(device),
			)
		input_dims[modality_name] = args.hidden_dim

	full_comb = "|".join([_group_name(g) for g in modality_groups])
	combination_to_index = {full_comb: 0}
	modality_combinations = [full_comb for _ in modality_combinations]
	data_dict["modality_comb"] = [
		combination_to_index[comb] if comb in combination_to_index else -1
		for comb in modality_combinations
	]

	train_idxs = [idx for idx in split_to_idxs["train"] if data_dict["modality_comb"][idx] != -1]
	valid_idxs = [idx for idx in split_to_idxs["val"] if data_dict["modality_comb"][idx] != -1]
	test_idxs = [idx for idx in split_to_idxs["test"] if data_dict["modality_comb"][idx] != -1]

	mc_num_to_mc = {v: k for k, v in combination_to_index.items()}
	mc_idx_dict = {
		mc_num_to_mc[mc_num]: list(
			np.where(np.array(data_dict["modality_comb"]) == mc_num)[0]
		)
		for mc_num in set(data_dict["modality_comb"])
		if mc_num != -1
	}

	n_labels = 5
	tqdm.write(
		f"DREAMT loader: train/val/test sizes = "
		f"{len(train_idxs)}/{len(valid_idxs)}/{len(test_idxs)}"
	)
	return (
		data_dict,
		encoder_dict,
		labels,
		train_idxs,
		valid_idxs,
		test_idxs,
		n_labels,
		input_dims,
		transforms,
		masks,
		observed_idx_arr,
		mc_idx_dict,
		mc_num_to_mc,
	)
