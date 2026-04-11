import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze modality search JSONL logs and rank token interaction priorities."
    )
    parser.add_argument(
        "--search_log",
        type=str,
        required=True,
        help="Path to modality_search JSONL log file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="Number of top tokens to print.",
    )
    parser.add_argument(
        "--trend_window",
        type=int,
        default=20,
        help="Number of trials in early/late windows for trend delta.",
    )
    parser.add_argument(
        "--include_failed",
        type=bool,
        default=False,
        help="Include failed trial records when scanning (normally false).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show best modality sets sorted by test accuracy (with F1 and kappa) instead of token-priority analysis.",
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Search log not found: {path}")
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


def parse_tokens(modality_text):
    text = str(modality_text).strip()
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


def mean_interaction_and_synergy(tokens, routing):
    if not isinstance(routing, list):
        return 0.0, 0.0
    n = len(tokens)
    if n == 0 or len(routing) <= n:
        return 0.0, 0.0
    token_weights = np.array(routing[:n], dtype=float)
    mean_interaction = float(np.mean(token_weights)) if token_weights.size > 0 else 0.0
    synergy_weight = float(routing[n]) if n < len(routing) else 0.0
    return mean_interaction, synergy_weight


def token_scaled_priority(tokens, routing):
    mean_interaction, synergy_weight = mean_interaction_and_synergy(tokens, routing)
    if mean_interaction <= 0.0:
        return {}
    eps = 1e-8
    out = {}
    for idx, tok in enumerate(tokens):
        tok_interaction = float(routing[idx])
        normalized = tok_interaction / (mean_interaction + eps)
        out[tok] = float(normalized * max(0.0, synergy_weight))
    return out


def build_trial_rows(rows, include_failed=False):
    trial_rows = []
    for r in rows:
        if r.get("record_type") != "trial_result":
            continue
        if not include_failed and r.get("status") != "completed":
            continue
        trial_rows.append(r)
    return trial_rows


def aggregate_token_stats(trial_rows):
    token_values = defaultdict(list)
    token_scores = defaultdict(list)
    token_kappa = defaultdict(list)
    token_test_acc = defaultdict(list)

    for row in trial_rows:
        tokens = parse_tokens(row.get("modality", ""))
        if not tokens:
            continue

        per_run = row.get("token_priority_scaled")
        if not isinstance(per_run, dict) or len(per_run) == 0:
            routing = row.get("routing_weight_mean_per_expert")
            if isinstance(routing, list):
                per_run = token_scaled_priority(tokens, routing)
            else:
                per_run = {}

        score = row.get("score")
        kappa = row.get("test_cohen_kappa")
        test_acc = row.get("test_acc")

        for tok in tokens:
            if tok in per_run and per_run[tok] is not None:
                token_values[tok].append(float(per_run[tok]))
            if score is not None:
                token_scores[tok].append(float(score))
            if kappa is not None:
                token_kappa[tok].append(float(kappa))
            if test_acc is not None:
                token_test_acc[tok].append(float(test_acc))

    stats = {}
    all_tokens = set(token_values) | set(token_scores) | set(token_kappa) | set(token_test_acc)
    for tok in sorted(all_tokens):
        vals = token_values.get(tok, [])
        scores = token_scores.get(tok, [])
        kappas = token_kappa.get(tok, [])
        accs = token_test_acc.get(tok, [])

        stats[tok] = {
            "runs": int(max(len(vals), len(scores), len(kappas), len(accs))),
            "priority_mean": float(np.mean(vals)) if len(vals) else 0.0,
            "priority_std": float(np.std(vals)) if len(vals) else 0.0,
            "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
            "kappa_mean": float(np.mean(kappas)) if len(kappas) else 0.0,
            "test_acc_mean": float(np.mean(accs)) if len(accs) else 0.0,
        }
    return stats


def compute_trend_deltas(trial_rows, trend_window):
    if len(trial_rows) == 0:
        return {}

    w = max(1, int(trend_window))
    early = trial_rows[:w]
    late = trial_rows[-w:]

    early_stats = aggregate_token_stats(early)
    late_stats = aggregate_token_stats(late)

    deltas = {}
    all_tokens = set(early_stats.keys()) | set(late_stats.keys())
    for tok in all_tokens:
        e = early_stats.get(tok, {})
        l = late_stats.get(tok, {})
        deltas[tok] = {
            "priority_delta": float(l.get("priority_mean", 0.0) - e.get("priority_mean", 0.0)),
            "score_delta": float(l.get("score_mean", 0.0) - e.get("score_mean", 0.0)),
            "kappa_delta": float(l.get("kappa_mean", 0.0) - e.get("kappa_mean", 0.0)),
            "test_acc_delta": float(l.get("test_acc_mean", 0.0) - e.get("test_acc_mean", 0.0)),
        }
    return deltas


def top_tokens(stats, top_k):
    items = list(stats.items())
    items.sort(
        key=lambda kv: (
            kv[1].get("priority_mean", 0.0),
            kv[1].get("score_mean", 0.0),
            kv[1].get("kappa_mean", 0.0),
        ),
        reverse=True,
    )
    return items[:top_k]


def print_table(top_rows, deltas):
    header = (
        f"{'Token':<8} {'Runs':>6} {'PriorityMean':>14} {'PriorityStd':>12} "
        f"{'ScoreMean':>10} {'KappaMean':>10} {'AccMean':>10} {'dPriority':>10}"
    )
    print(header)
    print("-" * len(header))
    for tok, s in top_rows:
        d = deltas.get(tok, {})
        print(
            f"{tok:<8} {s['runs']:>6d} {s['priority_mean']:>14.6f} {s['priority_std']:>12.6f} "
            f"{s['score_mean']:>10.4f} {s['kappa_mean']:>10.4f} {s['test_acc_mean']:>10.2f} "
            f"{d.get('priority_delta', 0.0):>10.6f}"
        )


def best_sets_by_accuracy(trial_rows):
    best_by_set = {}
    for row in trial_rows:
        if row.get("status") != "completed":
            continue
        modality = str(row.get("modality", "")).strip()
        if not modality:
            continue
        current = best_by_set.get(modality)
        acc = float(row.get("test_acc") or 0.0)
        f1 = float(row.get("test_f1_macro") or 0.0)
        kappa = float(row.get("test_cohen_kappa") or 0.0)
        if current is None:
            best_by_set[modality] = row
            continue
        curr_acc = float(current.get("test_acc") or 0.0)
        curr_f1 = float(current.get("test_f1_macro") or 0.0)
        curr_kappa = float(current.get("test_cohen_kappa") or 0.0)
        if (acc, f1, kappa) > (curr_acc, curr_f1, curr_kappa):
            best_by_set[modality] = row
    rows = list(best_by_set.values())
    rows.sort(
        key=lambda r: (
            float(r.get("test_acc") or 0.0),
            float(r.get("test_f1_macro") or 0.0),
            float(r.get("test_cohen_kappa") or 0.0),
        ),
        reverse=True,
    )
    return rows


def print_set_report(rows, top_k):
    header = (
        f"{'ModalitySet':<20},{'Size':>4},{'TestAcc':>10},{'TestF1':>10} "
        f"{'Kappa':>10},{'Score':>10},{'Seed':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows[:top_k]:
        modality = str(row.get("modality", ""))
        print(
            f"{modality:<20}, {len(modality):>4d}, "
            f"{float(row.get('test_acc') or 0.0):>10.2f},"
            f"{float(row.get('test_f1_macro') or 0.0):>10.2f},"
            f"{float(row.get('test_cohen_kappa') or 0.0):>10.4f}, "
            f"{float(row.get('score') or 0.0):>10.4f}, "
            f"{int(row.get('seed') or 0):>6d}"
        )


def main():
    args = parse_args()
    rows = load_jsonl(args.search_log)
    trial_rows = build_trial_rows(rows, include_failed=args.include_failed)

    if len(trial_rows) == 0:
        print("No trial_result rows found to analyze.")
        return

    if args.report:
        report_rows = best_sets_by_accuracy(trial_rows)
        print(f"Search log: {args.search_log}")
        print(f"Trial rows analyzed: {len(trial_rows)}")
        print(f"Unique modality sets: {len(report_rows)}")
        print_set_report(report_rows, args.top_k)
        return

    stats = aggregate_token_stats(trial_rows)
    deltas = compute_trend_deltas(trial_rows, args.trend_window)
    top_rows = top_tokens(stats, args.top_k)

    print(f"Search log: {args.search_log}")
    print(f"Trial rows analyzed: {len(trial_rows)}")
    print(f"Unique tokens: {len(stats)}")
    print_table(top_rows, deltas)


if __name__ == "__main__":
    main()
