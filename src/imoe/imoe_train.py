import torch
from tqdm import trange
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from copy import deepcopy
from datetime import datetime
from fvcore.nn import FlopCountAnalysis, parameter_count
import random
import time

from src.common.datasets.adni import load_and_preprocess_data_adni
from src.common.datasets.mimic import load_and_preprocess_data_mimic
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.dreamt import load_and_preprocess_data_dreamt
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.mosi import (
    load_and_preprocess_data_mosi,
    load_and_preprocess_data_mosi_regression,
)
from src.common.datasets.MultiModalDataset import create_loaders

from src.common.utils import (
    seed_everything,
    plot_total_loss_curves,
    plot_interaction_loss_curves,
    visualize_sample_weights,
    visualize_expert_logits,
    visualize_expert_logits_distribution,
    set_style,
)

from src.imoe.InteractionMoE import InteractionMoE
from src.imoe.InteractionMoERegression import InteractionMoERegression

set_style()


class ExpertLogitUncertaintyCNN(torch.nn.Module):
    """Predicts correctness probability from expert/fused logit matrices."""

    def __init__(self, base_channels=16):
        super().__init__()
        c = int(base_channels)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, c, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.head = torch.nn.Linear(c * 4, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = torch.sigmoid(self.head(x)).squeeze(1)
        return x


def _split_train_and_calibration_ids(train_ids, calibration_ratio, seed):
    train_ids = list(train_ids)
    ratio = float(calibration_ratio)
    if ratio <= 0.0 or len(train_ids) < 2:
        return train_ids, []
    shuffled = list(train_ids)
    random.Random(seed).shuffle(shuffled)
    cal_size = int(len(shuffled) * ratio)
    cal_size = max(1, min(cal_size, len(shuffled) - 1))
    calibration_ids = shuffled[:cal_size]
    train_ids_main = shuffled[cal_size:]
    return train_ids_main, calibration_ids


def _build_uncertainty_input(expert_outputs, fused_logits):
    # Shape: [B, 1, n_classes, n_experts + 1]
    expert_stack = torch.stack(expert_outputs, dim=2)
    fused_col = fused_logits.unsqueeze(2)
    return torch.cat([expert_stack, fused_col], dim=2).unsqueeze(1)


def _train_uncertainty_estimator(
    args,
    ensemble_model,
    encoder_dict,
    calibration_loader,
    device,
):
    if calibration_loader is None:
        return None, {}

    uncertainty_model = ExpertLogitUncertaintyCNN(
        base_channels=int(getattr(args, "uncertainty_base_channels", 16))
    ).to(device)
    optimizer = torch.optim.Adam(
        uncertainty_model.parameters(),
        lr=float(getattr(args, "uncertainty_lr", 1e-3)),
        weight_decay=float(getattr(args, "uncertainty_weight_decay", 0.0)),
    )
    criterion = torch.nn.BCELoss()
    epochs = int(getattr(args, "uncertainty_epochs", 5))

    ensemble_model.eval()
    for encoder in encoder_dict.values():
        encoder.eval()

    epoch_losses = []
    total_samples = 0
    for _ in range(max(1, epochs)):
        batch_losses = []
        for batch_samples, batch_labels, batch_mcs, batch_observed in calibration_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)

            with torch.no_grad():
                fusion_input = []
                for modality, samples in batch_samples.items():
                    fusion_input.append(encoder_dict[modality](samples))
                expert_outputs, _, fused_logits = ensemble_model.inference(fusion_input)
                _, preds = torch.max(fused_logits, 1)
                correctness = (preds == batch_labels).float()

            unc_input = _build_uncertainty_input(expert_outputs, fused_logits).detach()
            conf = uncertainty_model(unc_input)
            loss = criterion(conf, correctness)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            total_samples += int(batch_labels.shape[0])

        if len(batch_losses) > 0:
            epoch_losses.append(float(np.mean(batch_losses)))

    summary = {
        "uncertainty_calibration_bce": (
            float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else None
        ),
        "uncertainty_calibration_samples": int(total_samples),
    }
    return uncertainty_model, summary


def _per_class_accuracy(y_true, y_pred, n_labels):
    per_class = {}
    for class_idx in range(int(n_labels)):
        class_mask = y_true == class_idx
        class_count = int(np.sum(class_mask))
        if class_count == 0:
            per_class[str(class_idx)] = None
            continue
        class_acc = float(np.mean(y_pred[class_mask] == class_idx))
        per_class[str(class_idx)] = class_acc
    return per_class


def _label_names_for_dataset(data_name, n_labels):
    if data_name == "dreamt":
        return ["W", "N1", "N2", "N3", "REM"]
    if data_name == "mimic":
        return ["Negative", "Positive"]
    if data_name == "mosi":
        return ["Negative", "Positive"]
    if data_name == "adni":
        return ["CN", "MCI", "AD"]
    return [f"class_{idx}" for idx in range(int(n_labels))]


def _rename_per_class_accuracy_keys(per_class_acc, data_name, n_labels):
    label_names = _label_names_for_dataset(data_name, n_labels)
    named = {}
    for class_idx in range(int(n_labels)):
        key = str(class_idx)
        label = (
            label_names[class_idx]
            if class_idx < len(label_names)
            else f"class_{class_idx}"
        )
        named[label] = per_class_acc.get(key)
    return named


def _parse_uncertainty_levels(levels_arg):
    text = str(levels_arg).strip() if levels_arg is not None else ""
    if not text:
        return [0.5, 0.7, 0.8, 0.9]
    levels = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
        except ValueError:
            continue
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        levels.append(v)
    levels = sorted(set(levels))
    return levels if len(levels) > 0 else [0.5, 0.7, 0.8, 0.9]


def _safe_subset_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return None, None, None
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    kappa = cohen_kappa_score(y_true, y_pred)
    if kappa is None or not np.isfinite(kappa):
        kappa = None
    else:
        kappa = float(kappa)
    return acc, f1, kappa


def _resolve_results_log_path(args, fusion):
    explicit = str(getattr(args, "results_log", "")).strip()
    if explicit:
        return Path(explicit)
    return Path(f"./logs/imoe/{fusion}/{args.data}/{args.modality}_final_scores.jsonl")


def _append_live_results_log(args, fusion, payload):
    try:
        path = _resolve_results_log_path(args, fusion)
        path.parent.mkdir(exist_ok=True, parents=True)
        row = dict(payload)
        row["timestamp_utc"] = datetime.utcnow().isoformat(timespec="seconds")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as ex:
        print(f"[WARN] Failed to append live results log: {ex}")


def train_and_evaluate_imoe(args, seed, fusion_model, fusion):
    """Train and evaluate interaction MoE.

    Args:
        args (argparser.args): argument
        seed (int): random seed
        ensemble_model (nn.Module): ensemble model
        fusion (str): name of fusion method

    Raises:
        ValueError

    Returns:
        tuple: (best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc)
    """
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)
    num_modalities = len(args.modality)

    if args.data == "adni":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_adni(args)
    elif args.data == "mimic":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mimic(args)
    elif args.data == "mosi":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mosi(args)
    elif args.data == "sarcasm":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_sarcasm(args)
    elif args.data == "humor":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_humor(args)
    elif args.data == "enrico":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_enrico(args)
    elif args.data == "dreamt":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_dreamt(args)
    elif args.data == "mmimdb":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mmimdb(args)
    elif args.data == "mosi_regression":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mosi_regression(args)

    use_uncertainty_estimator = (
        bool(getattr(args, "use_uncertainty_estimator", False))
        and fusion == "transformer"
        and args.data not in ["mosi_regression", "mmimdb"]
    )

    train_ids_main, calibration_ids = _split_train_and_calibration_ids(
        train_ids,
        float(getattr(args, "uncertainty_calibration_ratio", 0.0)),
        seed,
    )
    if use_uncertainty_estimator and len(calibration_ids) == 0:
        print(
            "[WARN] Uncertainty estimator enabled but no calibration ids available; skipping uncertainty training."
        )
        use_uncertainty_estimator = False

    train_loader, val_loader, test_loader = create_loaders(
        data_dict,
        observed_idx_arr,
        labels,
        train_ids_main,
        valid_ids,
        test_ids,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        input_dims,
        transforms,
        masks,
        args.use_common_ids,
        dataset=args.data,
    )

    calibration_loader = None
    if use_uncertainty_estimator:
        calibration_loader, _, _ = create_loaders(
            data_dict,
            observed_idx_arr,
            labels,
            calibration_ids,
            calibration_ids,
            calibration_ids,
            args.batch_size,
            args.num_workers,
            args.pin_memory,
            input_dims,
            transforms,
            masks,
            args.use_common_ids,
            dataset=args.data,
        )

    ensemble_model = InteractionMoE(
        num_modalities=num_modalities,
        fusion_model=deepcopy(fusion_model),
        fusion_sparse=args.fusion_sparse,
        hidden_dim=args.hidden_dim,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
    ).to(device)

    if args.data == "mosi_regression":
        ensemble_model = InteractionMoERegression(
            num_modalities=num_modalities,
            fusion_model=deepcopy(fusion_model),
            fusion_sparse=args.fusion_sparse,
            hidden_dim=args.hidden_dim,
            hidden_dim_rw=args.hidden_dim_rw,
            num_layer_rw=args.num_layer_rw,
            temperature_rw=args.temperature_rw,
        ).to(device)

    params = list(ensemble_model.parameters()) + [
        param for encoder in encoder_dict.values() for param in encoder.parameters()
    ]

    optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.data in ["adni", "enrico", "dreamt", "mosi", "sarcasm", "humor"]:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.data == "mimic":
        criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))
    elif args.data == "mosi_regression":
        criterion = torch.nn.SmoothL1Loss()  # Regression
    elif args.data == "mmimdb":
        criterion = torch.nn.BCEWithLogitsLoss()

    if args.data == "mosi_regression":
        best_val_loss = 100000
    elif args.data == "mmimdb":
        best_val_f1 = 0
    else:
        best_val_acc = -1.0

    # Always keep a valid fallback checkpoint state.
    best_model_fus = deepcopy(ensemble_model.state_dict())
    best_model_enc = {
        modality: deepcopy(encoder.state_dict())
        for modality, encoder in encoder_dict.items()
    }
    if args.save:
        best_model_fus_cpu = {k: v.cpu() for k, v in best_model_fus.items()}
        best_model_enc_cpu = {
            modality: {k: v.cpu() for k, v in enc_state.items()}
            for modality, enc_state in best_model_enc.items()
        }

    if args.fusion_sparse:
        plotting_total_losses = {"task": [], "interaction": [], "gate": []}
    else:
        plotting_total_losses = {"task": [], "interaction": []}

    plotting_interaction_losses = {}
    for i in range(len(args.modality)):
        plotting_interaction_losses[f"uni_{i+1}"] = []
    plotting_interaction_losses[f"syn"] = []
    plotting_interaction_losses[f"red"] = []

    ############ efficiency
    train_time = 0
    ############ efficiency
    best_epoch_events = []

    for epoch in trange(args.train_epochs):
        ############ efficiency
        epoch_start_time = time.time()
        ############ efficiency

        ensemble_model.train()

        for encoder in encoder_dict.values():
            encoder.train()

        batch_task_losses = []
        if args.fusion_sparse:
            batch_gate_losses = []
        batch_interaction_losses = []

        num_interaction_experts = len(args.modality) + 2
        interaction_loss_sums = [0] * (num_interaction_experts)
        minibatch_count = len(train_loader)

        for batch_samples, batch_labels, batch_mcs, batch_observed in train_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()

            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            if args.fusion_sparse:
                _, _, outputs, interaction_losses, gate_losses = ensemble_model(
                    fusion_input
                )
            else:
                _, _, outputs, interaction_losses = ensemble_model(fusion_input)

            if args.data == "mosi_regression":
                task_loss = criterion(outputs, batch_labels.unsqueeze(1))
            else:
                task_loss = criterion(outputs, batch_labels)

            interaction_loss = sum(interaction_losses) / (len(args.modality) + 2)
            if args.fusion_sparse:
                gate_loss = torch.mean(torch.tensor(gate_losses))
                loss = (
                    task_loss
                    + args.interaction_loss_weight * interaction_loss
                    + args.gate_loss_weight * gate_loss
                )
            else:
                loss = task_loss + args.interaction_loss_weight * interaction_loss

            loss.backward()
            optimizer.step()

            batch_task_losses.append(task_loss.item())
            batch_interaction_losses.append(interaction_loss.item())
            if args.fusion_sparse:
                batch_gate_losses.append(gate_loss.item())

            for idx, loss in enumerate(interaction_losses):
                interaction_loss_sums[idx] += loss.item()

            if args.data == "enrico":
                torch.nn.utils.clip_grad_norm_(params, 1.0)

        ############ efficiency
        epoch_end_time = time.time()
        train_epoch_time = epoch_end_time - epoch_start_time
        train_time += train_epoch_time
        ############ efficiency

        plotting_total_losses["task"].append(np.mean(batch_task_losses))
        plotting_total_losses["interaction"].append(np.mean(batch_interaction_losses))
        if args.fusion_sparse:
            plotting_total_losses["gate"].append(np.mean(batch_gate_losses))

        for i in range(len(args.modality)):
            avg_loss = interaction_loss_sums[i] / minibatch_count
            plotting_interaction_losses[f"uni_{i+1}"].append(avg_loss)

        # For syn and red interaction losses
        plotting_interaction_losses["syn"].append(
            interaction_loss_sums[-2] / minibatch_count
        )
        plotting_interaction_losses["red"].append(
            interaction_loss_sums[-1] / minibatch_count
        )

        ensemble_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        val_losses = []

        with torch.no_grad():
            for batch_samples, batch_labels, batch_mcs, batch_observed in val_loader:
                batch_samples = {
                    k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
                }
                batch_labels = batch_labels.to(device, non_blocking=True)
                batch_mcs = batch_mcs.to(device, non_blocking=True)
                batch_observed = batch_observed.to(device, non_blocking=True)
                optimizer.zero_grad()

                fusion_input = []
                for i, (modality, samples) in enumerate(batch_samples.items()):
                    encoded_samples = encoder_dict[modality](samples)
                    fusion_input.append(encoded_samples)

                _, _, outputs = ensemble_model.inference(fusion_input)

                if args.data == "mosi_regression":
                    # if False:
                    val_loss = criterion(outputs, batch_labels.unsqueeze(1))
                    val_losses.append(val_loss.item())
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())

                else:
                    if args.data == "mmimdb":
                        val_loss = criterion(outputs, batch_labels.float())
                    else:
                        val_loss = criterion(outputs, batch_labels)
                    val_losses.append(val_loss.item())
                    if args.data == "mmimdb":
                        preds = torch.sigmoid(outputs).round()
                    else:
                        _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                    if args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                        all_probs.extend(
                            torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                            .cpu()
                            .numpy()
                        )
                    else:
                        probs = (
                            torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        )
                        all_probs.extend(probs)
                        if (
                            probs.shape[1] != n_labels
                        ):  # n_labels is the number of classes
                            raise ValueError("Incorrect output shape from the model")
        if args.data == "mosi_regression":
            val_loss = np.mean(val_losses)
            val_acc = accuracy_score(
                (np.array(all_preds) > 0), (np.array(all_labels) > 0)
            )
            print(
                f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}] Task Loss: {np.mean(val_losses):.2f} / Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch_events.append(
                    {
                        "epoch": int(epoch + 1),
                        "monitor": "val_loss",
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc * 100),
                    }
                )
                _append_live_results_log(
                    args,
                    fusion,
                    {
                        "record_type": "best_epoch",
                        "seed": int(seed),
                        **best_epoch_events[-1],
                    },
                )

                print(
                    f"[(**Best**) [Epoch {epoch+1}/{args.train_epochs}]  Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}"
                )

                best_model_fus = deepcopy(ensemble_model.state_dict())
                best_model_enc = {
                    modality: deepcopy(encoder.state_dict())
                    for modality, encoder in encoder_dict.items()
                }
                # Move the models to CPU for saving (only state_dict)
                if args.save:
                    best_model_fus_cpu = {k: v.cpu() for k, v in best_model_fus.items()}
                    best_model_enc_cpu = {
                        modality: {k: v.cpu() for k, v in enc_state.items()}
                        for modality, enc_state in best_model_enc.items()
                    }

        else:
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average="macro")
            val_auc = 0
            if args.data == "enrico":
                val_auc = roc_auc_score(
                    np.array(all_labels),
                    np.array(all_probs),
                    multi_class="ovo",
                    labels=list(range(n_labels)),
                )
            elif args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                val_auc = roc_auc_score(all_labels, all_probs)
            elif args.data == "mmimdb":
                val_auc = 0
            elif args.data == "adni":
                val_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

            print(
                f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}]  Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
            )

            if args.data == "mmimdb":
                # if False:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_acc = val_acc
                    best_val_auc = val_auc
                    best_epoch_events.append(
                        {
                            "epoch": int(epoch + 1),
                            "monitor": "val_f1",
                            "val_acc": float(val_acc * 100),
                            "val_f1": float(val_f1 * 100),
                            "val_auc": float(val_auc * 100),
                        }
                    )
                    _append_live_results_log(
                        args,
                        fusion,
                        {
                            "record_type": "best_epoch",
                            "seed": int(seed),
                            **best_epoch_events[-1],
                        },
                    )
                    print(
                        f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
                    )

                    best_model_fus = deepcopy(ensemble_model.state_dict())
                    best_model_enc = {
                        modality: deepcopy(encoder.state_dict())
                        for modality, encoder in encoder_dict.items()
                    }

                    if args.save:
                        best_model_fus_cpu = {
                            k: v.cpu() for k, v in best_model_fus.items()
                        }
                        best_model_enc_cpu = {
                            modality: {k: v.cpu() for k, v in enc_state.items()}
                            for modality, enc_state in best_model_enc.items()
                        }
            else:
                if val_acc > best_val_acc:
                    print(
                        f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
                    )
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_val_auc = val_auc
                    best_epoch_events.append(
                        {
                            "epoch": int(epoch + 1),
                            "monitor": "val_acc",
                            "val_acc": float(val_acc * 100),
                            "val_f1": float(val_f1 * 100),
                            "val_auc": float(val_auc * 100),
                        }
                    )
                    _append_live_results_log(
                        args,
                        fusion,
                        {
                            "record_type": "best_epoch",
                            "seed": int(seed),
                            **best_epoch_events[-1],
                        },
                    )
                    best_model_fus = deepcopy(ensemble_model.state_dict())
                    best_model_enc = {
                        modality: deepcopy(encoder.state_dict())
                        for modality, encoder in encoder_dict.items()
                    }
                    # Move the models to CPU for saving (only state_dict)
                    if args.save:
                        best_model_fus_cpu = {
                            k: v.cpu() for k, v in best_model_fus.items()
                        }
                        best_model_enc_cpu = {
                            modality: {k: v.cpu() for k, v in enc_state.items()}
                            for modality, enc_state in best_model_enc.items()
                        }
    ############ efficiency
    total_param = parameter_count(ensemble_model)[""]
    # flop = FlopCountAnalysis(ensemble_model, fusion_input)
    total_flop = 0
    ############ efficiency

    plot_total_loss_curves(
        args,
        plotting_total_losses=plotting_total_losses,
        framework="imoe",
        fusion=fusion,
    )

    plot_interaction_loss_curves(
        args,
        plotting_interaction_losses=plotting_interaction_losses,
        framework="imoe",
        fusion=fusion,
    )
    # Save the best model
    if args.save:
        Path("./saves").mkdir(exist_ok=True, parents=True)
        Path(f"./saves/imoe/{fusion}/{args.data}").mkdir(exist_ok=True, parents=True)

        if args.data == "mmimdb":
            save_path = f"./saves/imoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_f1_{best_val_f1:.2f}.pth"
        elif args.data == "mosi_regression":
            save_path = f"./saves/imoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_loss_{best_val_loss:.2f}.pth"
        else:
            save_path = f"./saves/imoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_acc_{best_val_acc:.2f}.pth"
        torch.save(
            {"ensemble_model": best_model_fus_cpu, "encoder_dict": best_model_enc_cpu},
            save_path,
        )

        print(f"Best model saved to {save_path}")

    # Load best model for test evaluation
    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()

    ensemble_model.load_state_dict(best_model_fus)
    ensemble_model.eval()

    uncertainty_model = None
    uncertainty_summary = {}
    if use_uncertainty_estimator:
        uncertainty_model, uncertainty_summary = _train_uncertainty_estimator(
            args,
            ensemble_model,
            encoder_dict,
            calibration_loader,
            device,
        )
        if uncertainty_model is not None:
            uncertainty_model.eval()

    all_preds = []
    all_labels = []
    all_ids = []
    all_probs = []
    all_uncertainty_probs = []
    all_uncertainty_targets = []
    test_losses = []
    all_routing_weights = []
    num_experts = len(args.modality) + 2
    all_expert_outputs = [[] for _ in range(num_experts)]

    ############ efficiency
    infer_time = 0
    ############ efficiency

    with torch.no_grad():
        ############ efficiency
        epoch_start_time = time.time()
        ############ efficiency

        for (
            batch_samples,
            batch_ids,
            batch_labels,
            batch_mcs,
            batch_observed,
        ) in test_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()

            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            expert_outputs, routing_weights, outputs = ensemble_model.inference(
                fusion_input
            )

            for expert_idx in range(num_experts):
                all_expert_outputs[expert_idx].extend(
                    expert_outputs[expert_idx].cpu().numpy()
                )

            all_routing_weights.extend(routing_weights.cpu().numpy())

            if args.data == "mosi_regression":
                all_preds.extend(outputs.squeeze().cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

            else:
                if args.data == "mmimdb":
                    preds = torch.sigmoid(outputs).round()
                else:
                    _, preds = torch.max(outputs, 1)

                if uncertainty_model is not None and args.data != "mmimdb":
                    unc_input = _build_uncertainty_input(expert_outputs, outputs)
                    unc_prob = uncertainty_model(unc_input)
                    all_uncertainty_probs.extend(unc_prob.cpu().numpy())
                    all_uncertainty_targets.extend(
                        (preds == batch_labels).float().cpu().numpy()
                    )

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_ids.extend(batch_ids.cpu().numpy())

                if args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    )
                else:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    )

    ############ efficiency
    epoch_end_time = time.time()
    infer_epoch_time = epoch_end_time - epoch_start_time
    infer_time += infer_epoch_time
    ############ efficiency

    visualize_expert_logits(
        expert_outputs, routing_weights, outputs, args, framework="imoe", fusion=fusion
    )

    visualize_expert_logits_distribution(
        all_expert_outputs, args, framework="imoe", fusion=fusion
    )

    visualize_sample_weights(all_routing_weights, args, framework="imoe", fusion=fusion)

    if args.data == "mosi_regression":
        all_binary_preds = np.array(all_preds) > 0
        all_labels = np.array(all_labels) > 0
        test_acc = accuracy_score(all_binary_preds, all_labels)
        test_mae = mean_absolute_error(all_preds, all_labels)

        now = datetime.now()
        save_dir = Path(
            f"./outputs/imoe/{fusion}/{args.data}_{now.strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        save_dir.mkdir(exist_ok=True, parents=True)
        np.save(save_dir / "all_expert_outputs.npy", np.array(all_expert_outputs))
        np.save(save_dir / "all_routing_weights.npy", np.array(all_routing_weights))
        np.save(save_dir / "all_preds.npy", np.array(all_preds))
        np.save(save_dir / "all_labels.npy", np.array(all_labels))
        np.save(save_dir / "all_ids.npy", np.array(all_ids))

        return (
            best_val_loss,
            best_val_acc,
            test_acc,
            test_mae,
            train_time / args.train_epochs,
            infer_time,
            total_flop,
            total_param,
        )
    else:
        all_labels_arr = np.array(all_labels)
        all_preds_arr = np.array(all_preds)
        test_acc = accuracy_score(all_labels_arr, all_preds_arr)
        test_f1 = f1_score(all_labels_arr, all_preds_arr, average="macro")
        test_f1_micro = f1_score(all_labels_arr, all_preds_arr, average="micro")
        test_kappa = cohen_kappa_score(all_labels_arr, all_preds_arr)
        test_per_class_acc = _per_class_accuracy(
            all_labels_arr, all_preds_arr, n_labels
        )
        test_auc = 0
        if args.data == "enrico":
            test_auc = roc_auc_score(
                np.array(all_labels),
                np.array(all_probs),
                multi_class="ovo",
                labels=list(range(n_labels)),
            )
        elif args.data in ["mimic", "mosi", "sarcasm", "humor"]:
            test_auc = roc_auc_score(all_labels, all_probs)
        elif args.data == "mmimdb":
            test_auc = 0
        elif args.data == "adni":
            test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

        now = datetime.now()
        save_dir = Path(
            f"./outputs/imoe/{fusion}/{args.data}_{now.strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        save_dir.mkdir(exist_ok=True, parents=True)
        np.save(save_dir / "all_expert_outputs.npy", np.array(all_expert_outputs))
        np.save(save_dir / "all_routing_weights.npy", np.array(all_routing_weights))
        np.save(save_dir / "all_preds.npy", np.array(all_preds))
        np.save(save_dir / "all_labels.npy", np.array(all_labels))
        np.save(save_dir / "all_ids.npy", np.array(all_ids))

        uncertainty_test_bce = None
        uncertainty_test_auc = None
        uncertainty_test_accuracy = None
        uncertainty_threshold_metrics = []
        if uncertainty_model is not None and len(all_uncertainty_probs) > 0:
            probs = np.array(all_uncertainty_probs, dtype=np.float32)
            targets = np.array(all_uncertainty_targets, dtype=np.float32)
            probs = np.clip(probs, 1e-6, 1 - 1e-6)
            uncertainty_test_bce = float(
                -np.mean(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
            )
            if len(np.unique(targets)) > 1:
                uncertainty_test_auc = float(roc_auc_score(targets, probs))

            unc_pred = (probs >= 0.5).astype(np.float32)
            uncertainty_test_accuracy = float(np.mean(unc_pred == targets))

            eval_levels = _parse_uncertainty_levels(
                getattr(args, "uncertainty_eval_levels", "0.5,0.7,0.8,0.9")
            )
            for level in eval_levels:
                keep_mask = probs >= float(level)
                pass_count = int(np.sum(keep_mask))
                pass_ratio = float(pass_count / max(1, len(probs)))
                expected_acc = float(level * 100.0)
                row = {
                    "threshold": float(level),
                    "pass_count": pass_count,
                    "pass_ratio": pass_ratio,
                    "expected_accuracy": expected_acc,
                    "accuracy": None,
                    "f1_macro": None,
                    "kappa": None,
                    "accuracy_gap": None,
                }
                if pass_count > 0:
                    subset_labels = all_labels_arr[keep_mask]
                    subset_preds = all_preds_arr[keep_mask]
                    acc_sub, f1_sub, kappa_sub = _safe_subset_metrics(
                        subset_labels, subset_preds
                    )
                    row["accuracy"] = (
                        float(acc_sub * 100.0) if acc_sub is not None else None
                    )
                    row["f1_macro"] = (
                        float(f1_sub * 100.0) if f1_sub is not None else None
                    )
                    row["kappa"] = kappa_sub
                    row["accuracy_gap"] = (
                        float(row["accuracy"] - expected_acc)
                        if row["accuracy"] is not None
                        else None
                    )
                uncertainty_threshold_metrics.append(row)

            print("Uncertainty-threshold test metrics:")
            for row in uncertainty_threshold_metrics:
                if row["accuracy"] is None:
                    print(
                        f"  >= {row['threshold']:.2f}: pass={row['pass_count']} ({row['pass_ratio']*100:.2f}%), no samples"
                    )
                else:
                    print(
                        f"  >= {row['threshold']:.2f}: pass={row['pass_count']} ({row['pass_ratio']*100:.2f}%), "
                        f"acc={row['accuracy']:.2f}, f1={row['f1_macro']:.2f}, kappa={row['kappa']}"
                    )
            np.save(save_dir / "uncertainty_probs.npy", probs)
            np.save(save_dir / "uncertainty_targets.npy", targets)

        routing_weights_arr = np.array(all_routing_weights)
        routing_weight_mean_per_expert = None
        if routing_weights_arr.size > 0 and routing_weights_arr.ndim == 2:
            routing_weight_mean_per_expert = (
                np.mean(routing_weights_arr, axis=0).astype(float).tolist()
            )

        if getattr(args, "return_detailed_metrics", False):
            named_per_class_acc = _rename_per_class_accuracy_keys(
                test_per_class_acc, args.data, n_labels
            )
            detailed_metrics = {
                "test_cohen_kappa": float(test_kappa),
                "test_per_class_accuracy": named_per_class_acc,
                "routing_weight_mean_per_expert": routing_weight_mean_per_expert,
                "output_dir": str(save_dir),
                "train_ids": [int(i) for i in train_ids],
                "valid_ids": [int(i) for i in valid_ids],
                "test_ids": [int(i) for i in test_ids],
                "uncertainty_test_bce": uncertainty_test_bce,
                "uncertainty_test_auc": uncertainty_test_auc,
                "uncertainty_test_accuracy": uncertainty_test_accuracy,
                "uncertainty_threshold_metrics": uncertainty_threshold_metrics,
                "best_epoch_events": best_epoch_events,
                **uncertainty_summary,
            }
            return (
                best_val_acc,
                best_val_f1,
                best_val_auc,
                test_acc,
                test_f1,
                test_f1_micro,
                test_auc,
                train_time / args.train_epochs,
                infer_time,
                total_flop,
                total_param,
                detailed_metrics,
            )

        return (
            best_val_acc,
            best_val_f1,
            best_val_auc,
            test_acc,
            test_f1,
            test_f1_micro,
            test_auc,
            train_time / args.train_epochs,
            infer_time,
            total_flop,
            total_param,
        )
