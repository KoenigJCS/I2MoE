import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

from src.common.fusion_models.interpretcc import InterpretCC
from src.imoe.imoe_train import train_and_evaluate_imoe
from src.common.utils import setup_logger, str2bool


# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="iMoE-interpretcc")
    parser.add_argument("--data", type=str, default="adni")
    parser.add_argument(
        "--dreamt_data_dir", type=str, default="data/dreamt"
    )  # Root directory for DREAMT spectrogram dataset
    parser.add_argument(
        "--modality", type=str, default="IGCB"
    )  # I G C B for ADNI, L N C for MIMIC
    parser.add_argument(
        "--patch", type=str2bool, default=False
    )  # Use common ids across modalities
    parser.add_argument(
        "--num_patches", type=int, default=16
    )  # Use common ids across modalities
    parser.add_argument("--initial_filling", type=str, default="mean")  # None mean
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument(
        "--max_seeds",
        type=int,
        default=0,
        help="If > 0, only use the first N seeds from n_runs.",
    )
    parser.add_argument(
        "--results_log",
        type=str,
        default="",
        help="Optional JSONL file path to append final summary metrics.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4
    )  # Number of workers for DataLoader
    parser.add_argument(
        "--pin_memory", type=str2bool, default=True
    )  # Pin memory in DataLoader
    parser.add_argument(
        "--use_common_ids", type=str2bool, default=True
    )  # Use common ids across modalities
    parser.add_argument(
        "--save", type=str2bool, default=True
    )  # Use common ids across modalities
    parser.add_argument(
        "--debug", type=str2bool, default=False
    )  # Use common ids across modalities

    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--temperature_rw", type=float, default=1
    )  # Temperature of the reweighting model
    parser.add_argument(
        "--hidden_dim_rw", type=int, default=256
    )  # Hidden dimension of the reweighting model
    parser.add_argument(
        "--num_layer_rw", type=int, default=1
    )  # Number of layers of the reweighting model
    parser.add_argument("--interaction_loss_weight", type=float, default=1e-2)
    parser.add_argument(
        "--fusion_sparse", type=str2bool, default=False
    )  # Whether to include SMoE in Fusion Layer
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--num_layers_enc", type=int, default=1
    )  # Number of MLP layers for encoders
    parser.add_argument(
        "--num_layers_fus", type=int, default=1
    )  # Number of MLP layers for fusion model
    parser.add_argument(
        "--num_layers_pred", type=int, default=1
    )  # Number of MLP layers for fusion model

    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard", type=str2bool, default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)  # Number of Routers

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    args.return_detailed_metrics = True
    logger = setup_logger(
        f"./logs/imoe/interpretcc/{args.data}",
        f"{args.data}",
        f"{args.modality}.txt",
    )
    seeds = np.arange(args.n_runs)  # [0, 1, 2]
    if args.max_seeds > 0:
        seeds = seeds[: args.max_seeds]
    if len(seeds) == 0:
        raise ValueError("No seeds selected. Check n_runs/max_seeds.")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    num_modalities = num_modality = len(args.modality)

    results_log_path = (
        Path(args.results_log)
        if args.results_log
        else Path(
            f"./logs/imoe/interpretcc/{args.data}/{args.modality}_final_scores.jsonl"
        )
    )
    results_log_path.parent.mkdir(exist_ok=True, parents=True)

    def append_final_scores(payload):
        payload = dict(payload)
        payload["timestamp_utc"] = datetime.utcnow().isoformat(timespec="seconds")
        with open(results_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def append_generation_scores(payload):
        payload = dict(payload)
        payload["record_type"] = "generation"
        append_final_scores(payload)

    log_summary = "======================================================================================\n"

    model_kwargs = {
        "model": "Interaction-MoE-interpretcc",
        "temperature_rw": args.temperature_rw,
        "hidden_dim_rw": args.hidden_dim_rw,
        "num_layer_rw": args.num_layer_rw,
        "interaction_loss_weight": args.interaction_loss_weight,
        "modality": args.modality,
        "tau": args.tau,
        "hard": args.hard,
        "threshold": args.threshold,
        "initial_filling": args.initial_filling,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print("Modality:", args.modality)

    data_to_nlabels = {
        "adni": 3,
        "mimic": 2,
        "mmimdb": 23,
        "enrico": 20,
        "dreamt": 5,
        "mosi": 2,
        "mosi_regression": 1,
    }
    n_labels = data_to_nlabels[args.data]

    if args.data == "mosi_regression":
        val_losses = []
        val_accs = []
        test_accs = []
        test_maes = []
    else:
        val_accs = []
        val_f1s = []
        val_aucs = []
        test_accs = []
        test_f1s = []
        test_f1_micros = []
        test_aucs = []
    generation_logs = []

    ############ efficiency
    train_times = []
    infer_times = []
    flops = []
    params = []
    ############ efficiency

    if len(seeds) == 1:
        fusion_model = InterpretCC(
            num_classes=n_labels,
            num_modality=len(args.modality),
            input_dim=args.hidden_dim,
            dropout=args.dropout,
            tau=args.tau,
            hard=args.hard,
            threshold=args.threshold,
        ).to(device)

        if args.data == "mosi_regression":
            (
                val_loss,
                val_acc,
                test_acc,
                test_mae,
                train_time,
                infer_time,
                flop,
                param,
            ) = train_and_evaluate_imoe(args, args.seed, fusion_model, "interpretcc")
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            test_maes.append(test_mae)
            generation_payload = {
                "fusion": "interpretcc",
                "data": args.data,
                "modality": args.modality,
                "seed": int(args.seed),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc * 100),
                "test_acc": float(test_acc * 100),
                "test_mae": float(test_mae),
            }
        else:
            run_output = train_and_evaluate_imoe(args, args.seed, fusion_model, "interpretcc")
            if len(run_output) == 12:
                (
                    val_acc,
                    val_f1,
                    val_auc,
                    test_acc,
                    test_f1,
                    test_f1_micro,
                    test_auc,
                    train_time,
                    infer_time,
                    flop,
                    param,
                    run_details,
                ) = run_output
            else:
                (
                    val_acc,
                    val_f1,
                    val_auc,
                    test_acc,
                    test_f1,
                    test_f1_micro,
                    test_auc,
                    train_time,
                    infer_time,
                    flop,
                    param,
                ) = run_output
                run_details = {}
            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            val_aucs.append(val_auc)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            test_f1_micros.append(test_f1_micro)
            test_aucs.append(test_auc)
            generation_payload = {
                "fusion": "interpretcc",
                "data": args.data,
                "modality": args.modality,
                "seed": int(args.seed),
                "val_acc": float(val_acc * 100),
                "val_f1": float(val_f1 * 100),
                "val_auc": float(val_auc * 100),
                "test_acc": float(test_acc * 100),
                "test_f1_macro": float(test_f1 * 100),
                "test_f1_micro": float(test_f1_micro * 100),
                "test_auc": float(test_auc * 100),
                "test_cohen_kappa": run_details.get("test_cohen_kappa"),
                "test_per_class_accuracy": run_details.get("test_per_class_accuracy"),
            }
        ############ efficiency
        train_times.append(train_time)
        infer_times.append(infer_time)
        flops.append(flop)
        params.append(param)
        generation_logs.append(generation_payload)
        append_generation_scores(generation_payload)
        ############ efficiency
    else:
        for seed in seeds:
            fusion_model = InterpretCC(
                num_classes=n_labels,
                num_modality=len(args.modality),
                input_dim=args.hidden_dim,
                dropout=args.dropout,
                tau=args.tau,
                hard=args.hard,
                threshold=args.threshold,
            ).to(device)

            if args.data == "mosi_regression":
                (
                    val_loss,
                    val_acc,
                    test_acc,
                    test_mae,
                    train_time,
                    infer_time,
                    flop,
                    param,
                ) = train_and_evaluate_imoe(
                    args, args.seed, fusion_model, "interpretcc"
                )
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                test_accs.append(test_acc)
                test_maes.append(test_mae)
                generation_payload = {
                    "fusion": "interpretcc",
                    "data": args.data,
                    "modality": args.modality,
                    "seed": int(args.seed),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc * 100),
                    "test_acc": float(test_acc * 100),
                    "test_mae": float(test_mae),
                }

            else:
                run_output = train_and_evaluate_imoe(args, seed, fusion_model, "interpretcc")
                if len(run_output) == 12:
                    (
                        val_acc,
                        val_f1,
                        val_auc,
                        test_acc,
                        test_f1,
                        test_f1_micro,
                        test_auc,
                        train_time,
                        infer_time,
                        flop,
                        param,
                        run_details,
                    ) = run_output
                else:
                    (
                        val_acc,
                        val_f1,
                        val_auc,
                        test_acc,
                        test_f1,
                        test_f1_micro,
                        test_auc,
                        train_time,
                        infer_time,
                        flop,
                        param,
                    ) = run_output
                    run_details = {}

                val_accs.append(val_acc)
                val_f1s.append(val_f1)
                val_aucs.append(val_auc)
                test_accs.append(test_acc)
                test_f1s.append(test_f1)
                test_f1_micros.append(test_f1_micro)
                test_aucs.append(test_auc)
                generation_payload = {
                    "fusion": "interpretcc",
                    "data": args.data,
                    "modality": args.modality,
                    "seed": int(seed),
                    "val_acc": float(val_acc * 100),
                    "val_f1": float(val_f1 * 100),
                    "val_auc": float(val_auc * 100),
                    "test_acc": float(test_acc * 100),
                    "test_f1_macro": float(test_f1 * 100),
                    "test_f1_micro": float(test_f1_micro * 100),
                    "test_auc": float(test_auc * 100),
                    "test_cohen_kappa": run_details.get("test_cohen_kappa"),
                    "test_per_class_accuracy": run_details.get("test_per_class_accuracy"),
                }
            ############ efficiency
            train_times.append(train_time)
            infer_times.append(infer_time)
            flops.append(flop)
            params.append(param)
            generation_logs.append(generation_payload)
            append_generation_scores(generation_payload)
            ############ efficiency

    ############ efficiency
    mean_train_time = np.mean(train_times)
    variance_train_time = np.var(train_times)
    mean_infer_time = np.mean(infer_times)
    variance_infer_time = np.var(infer_times)
    mean_flop = np.mean(flops)
    variance_flop = np.var(flops)
    mean_gflop = np.mean(np.array(flops) / 1e9)
    variance_gflop = np.var(np.array(flops) / 1e9)
    mean_param = np.mean(params)
    variance_param = np.var(params)

    log_summary += "\n"
    log_summary += (
        f"Train one epoch time: {mean_train_time:.2f} ± {variance_train_time:.2f} "
    )
    log_summary += "\n"
    log_summary += (
        f"Inference one epoch time: {mean_infer_time:.2f} ± {variance_infer_time:.2f} "
    )
    log_summary += "\n"
    log_summary += f"flops: {mean_flop:,.0f} ± {variance_flop:,.0f} "
    log_summary += "\n"
    log_summary += f"gflops: {mean_gflop:.2f} ± {variance_gflop:.2f} "
    log_summary += "\n"
    log_summary += f"param: {mean_param:,.0f} ± {variance_param:,.0f} "
    log_summary += "\n"
    ############ efficiency
    if args.data == "mosi_regression":
        val_avg_acc = np.mean(val_accs) * 100
        val_std_acc = np.std(val_accs) * 100
        val_avg_loss = np.mean(val_losses)
        val_std_loss = np.std(val_losses)
        test_avg_acc = np.mean(test_accs) * 100
        test_std_acc = np.std(test_accs) * 100
        test_avg_mae = np.mean(test_maes)
        test_std_mae = np.std(test_maes)

        log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} "
        log_summary += f"[Val] Average Loss: {val_avg_loss:.2f} ± {val_std_loss:.2f} "
        log_summary += (
            f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f}  "
        )
        log_summary += (
            f"[Test] Mean Absolute Error: {test_avg_mae:.2f} ± {test_std_mae:.2f}  "
        )

        print(model_kwargs)
        print(
            f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average Loss: {val_avg_loss:.2f} ± {val_std_loss:.2f} "
        )
        print(f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} ")

    else:

        val_avg_acc = np.mean(val_accs) * 100
        val_std_acc = np.std(val_accs) * 100
        val_avg_f1 = np.mean(val_f1s) * 100
        val_std_f1 = np.std(val_f1s) * 100
        val_avg_auc = np.mean(val_aucs) * 100
        val_std_auc = np.std(val_aucs) * 100

        test_avg_acc = np.mean(test_accs) * 100
        test_std_acc = np.std(test_accs) * 100
        test_avg_f1 = np.mean(test_f1s) * 100
        test_std_f1 = np.std(test_f1s) * 100
        test_avg_f1_micro = np.mean(test_f1_micros) * 100
        test_std_f1_micro = np.std(test_f1_micros) * 100
        test_avg_auc = np.mean(test_aucs) * 100
        test_std_auc = np.std(test_aucs) * 100

        log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} "
        log_summary += f"[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} "
        log_summary += f"[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / "
        log_summary += (
            f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} "
        )
        log_summary += (
            f"[Test] Average F1 (Macro) Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} "
        )
        log_summary += f"[Test] Average F1 (Micro) Score: {test_avg_f1_micro:.2f} ± {test_std_f1_micro:.2f} "
        log_summary += f"[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f} "

        print(model_kwargs)
        print(
            f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} / Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}"
        )
        print(
            f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} / Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}"
        )

    logger.info(log_summary)

    if args.data == "mosi_regression":
        final_scores = {
            "fusion": "interpretcc",
            "data": args.data,
            "modality": args.modality,
            "seeds": [int(s) for s in seeds.tolist()],
            "val_acc_mean": float(val_avg_acc),
            "val_acc_std": float(val_std_acc),
            "val_loss_mean": float(val_avg_loss),
            "val_loss_std": float(val_std_loss),
            "test_acc_mean": float(test_avg_acc),
            "test_acc_std": float(test_std_acc),
            "test_mae_mean": float(test_avg_mae),
            "test_mae_std": float(test_std_mae),
        }
    else:
        final_scores = {
            "fusion": "interpretcc",
            "data": args.data,
            "modality": args.modality,
            "seeds": [int(s) for s in seeds.tolist()],
            "val_acc_mean": float(val_avg_acc),
            "val_acc_std": float(val_std_acc),
            "val_f1_mean": float(val_avg_f1),
            "val_f1_std": float(val_std_f1),
            "val_auc_mean": float(val_avg_auc),
            "val_auc_std": float(val_std_auc),
            "test_acc_mean": float(test_avg_acc),
            "test_acc_std": float(test_std_acc),
            "test_f1_macro_mean": float(test_avg_f1),
            "test_f1_macro_std": float(test_std_f1),
            "test_f1_micro_mean": float(test_avg_f1_micro),
            "test_f1_micro_std": float(test_std_f1_micro),
            "test_auc_mean": float(test_avg_auc),
            "test_auc_std": float(test_std_auc),
        }
    append_final_scores(final_scores)
    print(f"Final scores appended to: {results_log_path}")


if __name__ == "__main__":
    main()
