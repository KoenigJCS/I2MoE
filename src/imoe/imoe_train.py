import torch
from tqdm import trange
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    confusion_matrix,
)
from copy import deepcopy
from datetime import datetime
from fvcore.nn import FlopCountAnalysis, parameter_count
import time
import matplotlib.pyplot as plt

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

TB_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter

    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False

set_style()


def _save_confusion_matrix_figure(
    cm, save_path, title="Confusion Matrix", return_figure=False
):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    if return_figure:
        return fig
    plt.close(fig)
    return None


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
    if args.data == "dreamt" and "," in args.modality:
        num_modalities = len([g for g in args.modality.split(",") if g.strip()])
    else:
        num_modalities = len(args.modality)
    tb_writer = None
    use_tensorboard = bool(getattr(args, "use_tensorboard", True))
    if use_tensorboard and TB_AVAILABLE:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_log_dir = Path(
            getattr(args, "tensorboard_log_dir", f"runs/imoe/{fusion}/{args.data}")
        )
        tb_log_dir = (
            base_log_dir
            / f"mod_{args.modality}_seed_{seed}_{run_stamp}"
        )
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard logging enabled: {tb_log_dir}")
    elif use_tensorboard and not TB_AVAILABLE:
        print("TensorBoard logging requested, but tensorboard is not installed.")

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

    train_loader, val_loader, test_loader = create_loaders(
        data_dict,
        observed_idx_arr,
        labels,
        train_ids,
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
    if args.data in ["adni", "enrico", "mosi", "sarcasm", "humor"]:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.data == "dreamt":
        labels_np = np.asarray(labels)
        train_labels = labels_np[np.asarray(train_ids, dtype=np.int64)]
        class_counts = np.bincount(train_labels, minlength=n_labels)
        class_weights = np.zeros(n_labels, dtype=np.float32)
        present = class_counts > 0
        if np.any(present):
            # Inverse-frequency weighting over classes observed in the train split.
            class_weights[present] = train_labels.shape[0] / (
                np.sum(present) * class_counts[present]
            )
        else:
            class_weights += 1.0
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)
        print(f"DREAMT class counts (train): {class_counts.tolist()}")
        print(
            "DREAMT class weights: "
            f"{[round(float(w), 4) for w in class_weights_t.detach().cpu().tolist()]}"
        )
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
        best_val_acc = 0.0

    if args.fusion_sparse:
        plotting_total_losses = {"task": [], "interaction": [], "gate": []}
    else:
        plotting_total_losses = {"task": [], "interaction": []}

    plotting_interaction_losses = {}
    for i in range(num_modalities):
        plotting_interaction_losses[f"uni_{i+1}"] = []
    plotting_interaction_losses[f"syn"] = []
    plotting_interaction_losses[f"red"] = []

    ############ efficiency
    train_time = 0
    ############ efficiency

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

        num_interaction_experts = num_modalities + 2
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

            interaction_loss = sum(interaction_losses) / (num_modalities + 2)
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

        if tb_writer is not None:
            tb_writer.add_scalar(
                "train/task_loss", np.mean(batch_task_losses), epoch + 1
            )
            tb_writer.add_scalar(
                "train/interaction_loss", np.mean(batch_interaction_losses), epoch + 1
            )
            if args.fusion_sparse:
                tb_writer.add_scalar(
                    "train/gate_loss", np.mean(batch_gate_losses), epoch + 1
                )

        for i in range(num_modalities):
            avg_loss = interaction_loss_sums[i] / minibatch_count
            plotting_interaction_losses[f"uni_{i+1}"].append(avg_loss)
            if tb_writer is not None:
                tb_writer.add_scalar(
                    f"train/interaction_uni_{i+1}", avg_loss, epoch + 1
                )

        # For syn and red interaction losses
        syn_loss = interaction_loss_sums[-2] / minibatch_count
        red_loss = interaction_loss_sums[-1] / minibatch_count
        plotting_interaction_losses["syn"].append(syn_loss)
        plotting_interaction_losses["red"].append(red_loss)
        if tb_writer is not None:
            tb_writer.add_scalar("train/interaction_syn", syn_loss, epoch + 1)
            tb_writer.add_scalar("train/interaction_red", red_loss, epoch + 1)

        n_uni_to_print = min(3, num_modalities)
        uni_losses_to_print = [
            interaction_loss_sums[i] / minibatch_count for i in range(n_uni_to_print)
        ]
        uni_msg = ", ".join(
            [f"uni_{i+1}: {uni_losses_to_print[i]:.4f}" for i in range(n_uni_to_print)]
        )
        print(
            f"[Epoch {epoch+1}/{args.train_epochs}] Interaction Losses -> "
            f"{uni_msg}, syn: {syn_loss:.4f}, red: {red_loss:.4f}"
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
            if tb_writer is not None:
                tb_writer.add_scalar("val/loss", val_loss, epoch + 1)
                tb_writer.add_scalar("val/acc", val_acc, epoch + 1)
            print(
                f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}] Task Loss: {np.mean(val_losses):.2f} / Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc

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
            mean_val_loss = float(np.mean(val_losses))
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
            elif args.data == "dreamt":
                try:
                    val_auc = roc_auc_score(
                        np.array(all_labels),
                        np.array(all_probs),
                        multi_class="ovo",
                        labels=list(range(n_labels)),
                    )
                except ValueError:
                    val_auc = 0
            elif args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                val_auc = roc_auc_score(all_labels, all_probs)
            elif args.data == "mmimdb":
                val_auc = 0
            elif args.data == "adni":
                val_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

            val_cm = confusion_matrix(
                all_labels, all_preds, labels=list(range(n_labels))
            )
            cm_save_path = (
                Path("figures")
                / "imoe"
                / fusion
                / "confusion_matrix"
                / args.data
                / f"seed_{seed}_mod_{args.modality}_epoch_{epoch+1}.png"
            )
            cm_fig = _save_confusion_matrix_figure(
                val_cm,
                cm_save_path,
                title=f"Val CM ({args.data}) Epoch {epoch+1}",
                return_figure=(tb_writer is not None),
            )
            if tb_writer is not None and cm_fig is not None:
                tb_writer.add_figure("val/confusion_matrix", cm_fig, epoch + 1)
                plt.close(cm_fig)

            if tb_writer is not None:
                tb_writer.add_scalar("val/loss", mean_val_loss, epoch + 1)
                tb_writer.add_scalar("val/acc", val_acc, epoch + 1)
                tb_writer.add_scalar("val/f1_macro", val_f1, epoch + 1)
                tb_writer.add_scalar("val/auc", val_auc, epoch + 1)

            print(
                f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}]  Val Loss: {mean_val_loss:.2f}, Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
            )

            if args.data == "mmimdb":
                # if False:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_acc = val_acc
                    best_val_auc = val_auc
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

    all_preds = []
    all_labels = []
    all_ids = []
    all_probs = []
    test_losses = []
    all_routing_weights = []
    num_experts = num_modalities + 2
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

        if tb_writer is not None:
            tb_writer.add_scalar("test/acc", test_acc)
            tb_writer.add_scalar("test/mae", test_mae)
            tb_writer.flush()
            tb_writer.close()

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
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average="macro")
        test_f1_micro = f1_score(all_labels, all_preds, average="micro")
        if args.data == "enrico":
            test_auc = roc_auc_score(
                np.array(all_labels),
                np.array(all_probs),
                multi_class="ovo",
                labels=list(range(n_labels)),
            )
        elif args.data == "dreamt":
            try:
                test_auc = roc_auc_score(
                    np.array(all_labels),
                    np.array(all_probs),
                    multi_class="ovo",
                    labels=list(range(n_labels)),
                )
            except ValueError:
                test_auc = 0
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

        if tb_writer is not None:
            tb_writer.add_scalar("test/acc", test_acc)
            tb_writer.add_scalar("test/f1_macro", test_f1)
            tb_writer.add_scalar("test/f1_micro", test_f1_micro)
            tb_writer.add_scalar("test/auc", test_auc)
            tb_writer.flush()
            tb_writer.close()

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
