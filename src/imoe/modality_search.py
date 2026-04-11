import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from src.common.datasets.dreamt import TOKEN_TO_MODALITY
from src.common.fusion_models.interpretcc import InterpretCC
from src.imoe.imoe_train import train_and_evaluate_imoe
from src.common.utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable modality search for iMoE-InterpretCC")

    # Search control
    parser.add_argument(
        "--method",
        type=str,
        default="greedy_forward",
        choices=["greedy_forward", "random", "top_p"],
    )
    parser.add_argument("--search_space", type=str, default="ALL", help="Token set: ALL/*, comma list (A,B,C), or compact string (ABC).")
    parser.add_argument("--start_modality", type=str, default="", help="Optional starting modality set, e.g. FE or F,E.")
    parser.add_argument("--min_modalities", type=int, default=1)
    parser.add_argument("--max_modalities", type=int, default=6)
    parser.add_argument("--max_trials", type=int, default=40)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus cumulative probability threshold for token selection.")
    parser.add_argument("--top_p_temperature", type=float, default=1.0, help="Temperature for converting token priorities into sampling probabilities.")
    parser.add_argument("--top_p_max_retries", type=int, default=200, help="Retries to propose an unseen combination in top_p mode.")

    # Scoring weights
    parser.add_argument("--w_kappa", type=float, default=1.0)
    parser.add_argument("--w_val_acc", type=float, default=0.2)
    parser.add_argument("--w_test_acc", type=float, default=0.3)
    parser.add_argument("--w_synergy", type=float, default=0.2)
    parser.add_argument("--w_redundancy", type=float, default=0.2)

    # Logging
    parser.add_argument("--search_log", type=str, default="")
    parser.add_argument("--split_manifest", type=str, default="", help="Optional JSON file path for canonical train/val/test split manifest.")
    parser.add_argument("--top_k", type=int, default=10)

    # Train/Eval args (compatible with train_interpretcc.py)
    parser.add_argument("--data", type=str, default="dreamt")
    parser.add_argument("--dreamt_data_dir", type=str, default="data/dreamt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--max_seeds", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--use_common_ids", type=str2bool, default=True)
    parser.add_argument("--save", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)

    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature_rw", type=float, default=1.0)
    parser.add_argument("--hidden_dim_rw", type=int, default=256)
    parser.add_argument("--num_layer_rw", type=int, default=1)
    parser.add_argument("--interaction_loss_weight", type=float, default=1e-2)
    parser.add_argument("--fusion_sparse", type=str2bool, default=False)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers_enc", type=int, default=1)
    parser.add_argument("--num_layers_fus", type=int, default=1)
    parser.add_argument("--num_layers_pred", type=int, default=1)
    parser.add_argument("--patch", type=str2bool, default=False)
    parser.add_argument("--num_patches", type=int, default=16)

    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard", type=str2bool, default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)

    # DREAMT loader options
    parser.add_argument("--dreamt_max_files", type=int, default=0)
    parser.add_argument("--dreamt_segment_seconds", type=int, default=30)
    parser.add_argument("--dreamt_img_dim_x", type=int, default=128)
    parser.add_argument("--dreamt_img_dim_y", type=int, default=256)
    parser.add_argument("--dreamt_normalize_image", type=str2bool, default=False)

    return parser.parse_args()


def now_utc():
    return datetime.utcnow().isoformat(timespec="seconds")


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row["timestamp_utc"] = now_utc()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_modality_tokens(text):
    text = str(text).strip()
    if not text:
        return []
    if "," in text:
        parts = [p.strip().upper() for p in text.split(",") if p.strip()]
    else:
        parts = [c.upper() for c in text if c.strip()]
    out = []
    for tok in parts:
        if tok not in out:
            out.append(tok)
    return out


def resolve_search_tokens(args):
    raw = str(args.search_space).strip()
    if raw.upper() in {"ALL", "*"}:
        if args.data == "dreamt":
            return list(TOKEN_TO_MODALITY.keys())
        raise ValueError("search_space=ALL is only auto-defined for dreamt. Provide explicit tokens for other datasets.")
    toks = parse_modality_tokens(raw)
    if not toks:
        raise ValueError("No search tokens resolved from --search_space")
    return toks


def modality_key(tokens):
    return "".join(sorted(tokens))


def n_labels_for_data(data_name):
    mapping = {
        "adni": 3,
        "mimic": 2,
        "mmimdb": 23,
        "enrico": 20,
        "dreamt": 5,
        "mosi": 2,
        "mosi_regression": 1,
    }
    if data_name not in mapping:
        raise ValueError(f"Unsupported data for search: {data_name}")
    return mapping[data_name]


def _normalize_split_ids(ids):
    if ids is None:
        return []
    return sorted(int(x) for x in ids)


def _extract_splits_from_result(result):
    return {
        "train_ids": _normalize_split_ids(result.get("train_ids")),
        "valid_ids": _normalize_split_ids(result.get("valid_ids")),
        "test_ids": _normalize_split_ids(result.get("test_ids")),
    }


def _splits_available(splits):
    return all(isinstance(splits.get(k), list) and len(splits.get(k)) > 0 for k in ["train_ids", "valid_ids", "test_ids"])


def _same_splits(a, b):
    return (
        a.get("train_ids", []) == b.get("train_ids", [])
        and a.get("valid_ids", []) == b.get("valid_ids", [])
        and a.get("test_ids", []) == b.get("test_ids", [])
    )


def _save_split_manifest(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_split_manifest(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def score_trial(result, combo_len, args):
    kappa = float(result.get("test_cohen_kappa") or 0.0)
    val_acc = float(result.get("val_acc") or 0.0) / 100.0
    test_acc = float(result.get("test_acc") or 0.0) / 100.0

    routing = result.get("routing_weight_mean_per_expert")
    synergy_w = 0.0
    redundancy_w = 0.0
    if isinstance(routing, list):
        syn_idx = combo_len
        red_idx = combo_len + 1
        if syn_idx < len(routing):
            synergy_w = float(routing[syn_idx])
        if red_idx < len(routing):
            redundancy_w = float(routing[red_idx])

    score = (
        args.w_kappa * kappa
        + args.w_val_acc * val_acc
        + args.w_test_acc * test_acc
        + args.w_synergy * synergy_w
        - args.w_redundancy * redundancy_w
    )

    return float(score), float(synergy_w), float(redundancy_w)


def _mean_interaction_and_synergy(tokens, routing):
    if not isinstance(routing, list):
        return 0.0, 0.0
    n = len(tokens)
    if n == 0 or len(routing) <= n:
        return 0.0, 0.0
    token_weights = np.array(routing[:n], dtype=float)
    mean_interaction = float(np.mean(token_weights)) if token_weights.size > 0 else 0.0
    synergy_weight = float(routing[n]) if n < len(routing) else 0.0
    return mean_interaction, synergy_weight


def build_token_priority_stats(done_rows, search_tokens):
    stats = {
        tok: {
            "sum_scaled_priority": 0.0,
            "count": 0,
            "mean_scaled_priority": 0.0,
        }
        for tok in search_tokens
    }
    eps = 1e-8

    for row in done_rows.values():
        if row.get("status") != "completed":
            continue
        tokens = parse_modality_tokens(row.get("modality", ""))
        routing = row.get("routing_weight_mean_per_expert")
        if not isinstance(routing, list):
            continue
        if len(tokens) == 0 or len(routing) < len(tokens):
            continue

        mean_interaction, synergy_weight = _mean_interaction_and_synergy(tokens, routing)
        if mean_interaction <= 0.0:
            continue

        for idx, tok in enumerate(tokens):
            if tok not in stats:
                continue
            token_interaction = float(routing[idx])
            normalized = token_interaction / (mean_interaction + eps)
            scaled_priority = normalized * max(0.0, synergy_weight)
            stats[tok]["sum_scaled_priority"] += float(scaled_priority)
            stats[tok]["count"] += 1

    for tok in search_tokens:
        if stats[tok]["count"] > 0:
            stats[tok]["mean_scaled_priority"] = (
                stats[tok]["sum_scaled_priority"] / stats[tok]["count"]
            )
        else:
            stats[tok]["mean_scaled_priority"] = 0.0

    return stats


def _weighted_choice(items, weights, rng):
    total = float(sum(weights))
    if total <= 0.0:
        return items[rng.randrange(len(items))]
    r = rng.random() * total
    c = 0.0
    for item, w in zip(items, weights):
        c += float(w)
        if r <= c:
            return item
    return items[-1]


def nucleus_sample_token(candidates, token_stats, top_p, temperature, rng):
    if len(candidates) == 1:
        return candidates[0]

    raw = np.array(
        [
            float(token_stats.get(tok, {}).get("mean_scaled_priority", 0.0))
            for tok in candidates
        ],
        dtype=float,
    )

    # Encourage outliers above average interaction while staying numerically stable.
    raw = np.clip(raw, a_min=0.0, a_max=None)
    if np.allclose(raw, 0.0):
        probs = np.ones_like(raw) / len(raw)
    else:
        temp = max(float(temperature), 1e-6)
        logits = raw / temp
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cum = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cum, float(top_p), side="left")) + 1
    cutoff = max(1, min(cutoff, len(candidates)))
    keep_idx = order[:cutoff]

    kept_tokens = [candidates[int(i)] for i in keep_idx]
    kept_probs = [float(probs[int(i)]) for i in keep_idx]
    return _weighted_choice(kept_tokens, kept_probs, rng)


def propose_top_p_combo(
    search_tokens,
    done_set,
    min_size,
    max_size,
    token_stats,
    top_p,
    temperature,
    max_retries,
    rng,
):
    max_size = min(max_size, len(search_tokens))
    tries = 0
    while tries < max_retries:
        tries += 1
        size = rng.randint(min_size, max_size)
        combo = []
        remaining = list(search_tokens)
        while len(combo) < size and remaining:
            next_tok = nucleus_sample_token(
                remaining,
                token_stats,
                top_p,
                temperature,
                rng,
            )
            combo.append(next_tok)
            remaining.remove(next_tok)
        key = modality_key(combo)
        if key not in done_set:
            return sorted(combo)
    return None


def run_interpretcc_eval(args, modality_str, seed):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    n_labels = n_labels_for_data(args.data)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.modality = modality_str
    eval_args.seed = int(seed)
    eval_args.return_detailed_metrics = True

    fusion_model = InterpretCC(
        num_classes=n_labels,
        num_modality=len(modality_str),
        input_dim=args.hidden_dim,
        dropout=args.dropout,
        tau=args.tau,
        hard=args.hard,
        threshold=args.threshold,
    ).to(device)

    output = train_and_evaluate_imoe(eval_args, eval_args.seed, fusion_model, "interpretcc")
    if len(output) == 12:
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
            details,
        ) = output
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
        ) = output
        details = {}

    trial = {
        "val_acc": float(val_acc * 100),
        "val_f1": float(val_f1 * 100),
        "val_auc": float(val_auc * 100),
        "test_acc": float(test_acc * 100),
        "test_f1_macro": float(test_f1 * 100),
        "test_f1_micro": float(test_f1_micro * 100),
        "test_auc": float(test_auc * 100),
        "train_time_per_epoch": float(train_time),
        "infer_time": float(infer_time),
        "flop": float(flop),
        "param": float(param),
        "test_cohen_kappa": details.get("test_cohen_kappa"),
        "test_per_class_accuracy": details.get("test_per_class_accuracy"),
        "routing_weight_mean_per_expert": details.get("routing_weight_mean_per_expert"),
        "output_dir": details.get("output_dir"),
        "train_ids": details.get("train_ids"),
        "valid_ids": details.get("valid_ids"),
        "test_ids": details.get("test_ids"),
    }
    return trial


def completed_trials_by_combo(rows):
    done = {}
    for row in rows:
        if row.get("record_type") != "trial_result":
            continue
        if row.get("status") != "completed":
            continue
        combo = str(row.get("modality", ""))
        if combo:
            done[combo] = row
    return done


def latest_state(rows):
    state = None
    for row in rows:
        if row.get("record_type") == "search_state":
            state = row
    return state


def propose_random_combo(search_tokens, done_set, min_size, max_size, rng):
    max_size = min(max_size, len(search_tokens))
    tries = 0
    while tries < 5000:
        tries += 1
        size = rng.randint(min_size, max_size)
        combo = sorted(rng.sample(search_tokens, size))
        key = modality_key(combo)
        if key not in done_set:
            return combo
    return None


def greedy_candidates(current, search_tokens):
    cur_set = set(current)
    out = []
    for tok in search_tokens:
        if tok in cur_set:
            continue
        cand = sorted(current + [tok])
        out.append(cand)
    return out


def run_search(args):
    rng = random.Random(args.random_seed)
    np.random.seed(args.random_seed)

    search_tokens = resolve_search_tokens(args)
    if args.min_modalities < 1:
        raise ValueError("--min_modalities must be >= 1")
    if args.max_modalities < args.min_modalities:
        raise ValueError("--max_modalities must be >= --min_modalities")

    default_log = f"./logs/modality_search/{args.data}/{args.method}_{args.search_space}.jsonl"
    search_log = args.search_log if args.search_log else default_log
    split_manifest_path = (
        args.split_manifest
        if args.split_manifest
        else f"./logs/modality_search/{args.data}/{args.method}_{args.search_space}_splits.json"
    )

    rows = load_jsonl(search_log) if args.resume else []
    done = completed_trials_by_combo(rows)
    done_set = set(done.keys())
    manifest_obj = _load_split_manifest(split_manifest_path) if args.resume else None
    expected_splits = None
    if manifest_obj:
        expected_splits = {
            "train_ids": _normalize_split_ids(manifest_obj.get("train_ids", [])),
            "valid_ids": _normalize_split_ids(manifest_obj.get("valid_ids", [])),
            "test_ids": _normalize_split_ids(manifest_obj.get("test_ids", [])),
        }

    state = latest_state(rows) if args.resume else None
    if state and state.get("method") == "greedy_forward":
        current = parse_modality_tokens(state.get("current_modality", ""))
    else:
        current = parse_modality_tokens(args.start_modality)

    start_record = {
        "record_type": "search_start",
        "method": args.method,
        "data": args.data,
        "search_space": "".join(search_tokens),
        "min_modalities": int(args.min_modalities),
        "max_modalities": int(args.max_modalities),
        "max_trials": int(args.max_trials),
        "resume": bool(args.resume),
        "already_completed": int(len(done_set)),
        "split_manifest": split_manifest_path,
    }
    append_jsonl(search_log, start_record)

    trial_counter = 0
    while trial_counter < args.max_trials:
        if args.method == "random":
            combo = propose_random_combo(
                search_tokens,
                done_set,
                args.min_modalities,
                args.max_modalities,
                rng,
            )
            if combo is None:
                break
        elif args.method == "top_p":
            token_stats = build_token_priority_stats(done, search_tokens)
            combo = propose_top_p_combo(
                search_tokens,
                done_set,
                args.min_modalities,
                args.max_modalities,
                token_stats,
                args.top_p,
                args.top_p_temperature,
                args.top_p_max_retries,
                rng,
            )
            if combo is None:
                break
            append_jsonl(
                search_log,
                {
                    "record_type": "token_priority_snapshot",
                    "method": "top_p",
                    "completed_trials": int(len(done_set)),
                    "token_priority": {
                        tok: float(token_stats[tok]["mean_scaled_priority"])
                        for tok in search_tokens
                    },
                },
            )
        else:
            if not current:
                combo = [search_tokens[0]]
                if modality_key(combo) in done_set:
                    combo = propose_random_combo(
                        search_tokens,
                        done_set,
                        args.min_modalities,
                        args.max_modalities,
                        rng,
                    )
                    if combo is None:
                        break
            else:
                if len(current) >= args.max_modalities:
                    break
                candidates = greedy_candidates(current, search_tokens)
                candidates = [c for c in candidates if len(c) <= args.max_modalities]
                if not candidates:
                    break

                # Evaluate missing candidates for this expansion step.
                step_results = []
                for cand in candidates:
                    if trial_counter >= args.max_trials:
                        break
                    key = modality_key(cand)
                    if key in done_set:
                        step_results.append(done[key])
                        continue

                    seed = args.seed + trial_counter
                    start_row = {
                        "record_type": "trial_start",
                        "method": args.method,
                        "modality": key,
                        "seed": int(seed),
                    }
                    append_jsonl(search_log, start_row)

                    try:
                        result = run_interpretcc_eval(args, key, seed)
                        trial_splits = _extract_splits_from_result(result)
                        if _splits_available(trial_splits):
                            if expected_splits is None:
                                expected_splits = trial_splits
                                manifest_obj = {
                                    "created_utc": now_utc(),
                                    "data": args.data,
                                    "method": args.method,
                                    "search_space": "".join(search_tokens),
                                    **trial_splits,
                                }
                                _save_split_manifest(split_manifest_path, manifest_obj)
                            elif not _same_splits(expected_splits, trial_splits):
                                raise ValueError("Split mismatch against canonical manifest for fair comparison.")
                        score, synergy_w, redundancy_w = score_trial(result, len(cand), args)
                        result_row = {
                            "record_type": "trial_result",
                            "status": "completed",
                            "method": args.method,
                            "data": args.data,
                            "modality": key,
                            "seed": int(seed),
                            "score": score,
                            "synergy_weight": synergy_w,
                            "redundancy_weight": redundancy_w,
                            **result,
                        }
                    except Exception as ex:
                        result_row = {
                            "record_type": "trial_result",
                            "status": "failed",
                            "method": args.method,
                            "data": args.data,
                            "modality": key,
                            "seed": int(seed),
                            "error": str(ex),
                        }

                    append_jsonl(search_log, result_row)
                    trial_counter += 1
                    if result_row.get("status") == "completed":
                        done[key] = result_row
                        done_set.add(key)
                        step_results.append(result_row)

                if not step_results:
                    break

                step_results = [r for r in step_results if r.get("status") == "completed"]
                if not step_results:
                    break
                best = max(step_results, key=lambda r: float(r.get("score", -1e9)))
                current = parse_modality_tokens(best["modality"])

                append_jsonl(
                    search_log,
                    {
                        "record_type": "search_state",
                        "method": "greedy_forward",
                        "current_modality": best["modality"],
                        "current_score": float(best.get("score", 0.0)),
                        "completed_trials": int(len(done_set)),
                    },
                )
                continue

        key = modality_key(combo)
        if key in done_set:
            trial_counter += 1
            continue

        seed = args.seed + trial_counter
        append_jsonl(
            search_log,
            {
                "record_type": "trial_start",
                "method": args.method,
                "modality": key,
                "seed": int(seed),
            },
        )

        try:
            result = run_interpretcc_eval(args, key, seed)
            trial_splits = _extract_splits_from_result(result)
            if _splits_available(trial_splits):
                if expected_splits is None:
                    expected_splits = trial_splits
                    manifest_obj = {
                        "created_utc": now_utc(),
                        "data": args.data,
                        "method": args.method,
                        "search_space": "".join(search_tokens),
                        **trial_splits,
                    }
                    _save_split_manifest(split_manifest_path, manifest_obj)
                elif not _same_splits(expected_splits, trial_splits):
                    raise ValueError("Split mismatch against canonical manifest for fair comparison.")
            score, synergy_w, redundancy_w = score_trial(result, len(combo), args)

            token_priorities = None
            if args.method == "top_p":
                mean_interaction, synergy_for_scale = _mean_interaction_and_synergy(
                    parse_modality_tokens(key),
                    result.get("routing_weight_mean_per_expert"),
                )
                token_priorities = {}
                routing = result.get("routing_weight_mean_per_expert")
                if isinstance(routing, list) and mean_interaction > 0.0:
                    for idx, tok in enumerate(parse_modality_tokens(key)):
                        token_priorities[tok] = float(
                            (float(routing[idx]) / mean_interaction)
                            * max(0.0, synergy_for_scale)
                        )

            result_row = {
                "record_type": "trial_result",
                "status": "completed",
                "method": args.method,
                "data": args.data,
                "modality": key,
                "seed": int(seed),
                "score": score,
                "synergy_weight": synergy_w,
                "redundancy_weight": redundancy_w,
                "token_priority_scaled": token_priorities,
                **result,
            }
        except Exception as ex:
            result_row = {
                "record_type": "trial_result",
                "status": "failed",
                "method": args.method,
                "data": args.data,
                "modality": key,
                "seed": int(seed),
                "error": str(ex),
            }

        append_jsonl(search_log, result_row)
        trial_counter += 1
        if result_row.get("status") == "completed":
            done[key] = result_row
            done_set.add(key)

    completed = [r for r in done.values() if r.get("status") == "completed"]
    completed = sorted(completed, key=lambda r: float(r.get("score", -1e9)), reverse=True)

    summary = {
        "record_type": "search_summary",
        "method": args.method,
        "data": args.data,
        "completed_trials": int(len(completed)),
        "top_k": int(args.top_k),
        "top_results": completed[: args.top_k],
    }
    append_jsonl(search_log, summary)

    print(f"Search log: {search_log}")
    print(f"Completed trials: {len(completed)}")
    for idx, row in enumerate(completed[: args.top_k], start=1):
        print(
            f"#{idx} modality={row.get('modality')} score={row.get('score'):.4f} "
            f"kappa={row.get('test_cohen_kappa')} test_acc={row.get('test_acc'):.2f}"
        )


if __name__ == "__main__":
    run_search(parse_args())
