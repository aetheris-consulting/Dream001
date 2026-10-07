#!/usr/bin/env python3
"""
Apply PREREGISTRATION_D8.md mechanically to a judged run.

  # 1. blind annotation sample (no flags, no judge labels in the file)
  python scripts/score_preregistered.py runs/<run> --n 757 --n-calibration 60 --export-sample
  # 2. after the sample is annotated into annotation_sample.jsonl ("hallucinated": bool)
  python scripts/score_preregistered.py runs/<run> --n 757 --n-calibration 60

Writes PREREG_RESULT.md and prereg_result.json into the run directory.
"""
import sys, os, json, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dream001 import RegimeConfig
from dream001.report import compute_metrics, verdict, LIFT_THRESHOLD, P_THRESHOLD
from run_truthfulqa import load_items

SAMPLE_SIZE, SAMPLE_SEED = 60, 1
AGREEMENT_MIN = 0.80
MIN_FLAGS = 30


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def export_sample(run_dir, evalset, items):
    idxs = sorted(random.Random(SAMPLE_SEED).sample(range(len(items)), SAMPLE_SIZE))
    path = os.path.join(run_dir, "annotation_sample_blind.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for i in idxs:
            ref = evalset[i]
            f.write(json.dumps({"idx": i, "question": ref["question"],
                                "correct_answers": ref["correct_answers"],
                                "incorrect_answers": ref["incorrect_answers"],
                                "generation": items[i]["generation"]}) + "\n")
    print(f"wrote {path} -- contains NO flags and NO judge labels")


def _metrics_block(m, v):
    return {"metrics": m.to_dict(), "verdict": v.to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--n-calibration", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-sample", action="store_true")
    a = ap.parse_args()

    evalset = load_items(a.n + a.n_calibration, seed=a.seed)[a.n_calibration:]
    items = _jsonl(os.path.join(a.run_dir, "items.jsonl"))
    if a.export_sample:
        return export_sample(a.run_dir, evalset, items)

    judge = {j["idx"]: j for j in _jsonl(os.path.join(a.run_dir, "judge_labels.jsonl"))}
    ann = {r["idx"]: bool(r["hallucinated"])
           for r in _jsonl(os.path.join(a.run_dir, "annotation_sample.jsonl"))}
    if set(judge) != {r["idx"] for r in items}:
        sys.exit("judge labels do not cover the run exactly")

    # --- judge validity gate ---------------------------------------------
    agree = sum(judge[i]["hallucinated"] == h for i, h in ann.items())
    agreement = agree / len(ann)
    judge_valid = len(ann) == SAMPLE_SIZE and agreement >= AGREEMENT_MIN

    flags = [r["flagged"] for r in items]
    y_judge = [judge[r["idx"]]["hallucinated"] for r in items]
    y_f1 = [r["label"]["hallucinated"] for r in items]

    m1 = compute_metrics(flags, y_judge); v1 = verdict(m1)
    m2 = compute_metrics(flags, y_f1); v2 = verdict(m2)

    cfg = RegimeConfig.short_sequence()
    floor = cfg.band_min_run + cfg.min_hold_run + 2
    cap = [k for k, r in enumerate(items) if r["n_tokens"] >= floor]
    mc = compute_metrics([flags[k] for k in cap], [y_judge[k] for k in cap])

    # --- decision rule, verbatim from the pre-registration ---------------
    if not judge_valid:
        decision = "UNRESOLVED (labels)"
    elif m1.n_flagged < MIN_FLAGS:
        decision = "UNDERPOWERED"
    elif v1.passed and mc.lift > LIFT_THRESHOLD and mc.permutation_p < P_THRESHOLD:
        decision = "SUPPORTED"
    elif m1.lift <= 1.0:
        decision = "REFUTED (on this testbed)"
    else:
        decision = "NOT SUPPORTED"

    judge_counts = {}
    for j in judge.values():
        judge_counts[j["judge"]] = judge_counts.get(j["judge"], 0) + 1

    res = {"decision": decision,
           "judge_validity": {"sample": len(ann), "agreement": agreement,
                              "threshold": AGREEMENT_MIN, "valid": judge_valid},
           "judge_counts": judge_counts,
           "primary_judge": _metrics_block(m1, v1),
           "capable_judge": {"floor": floor, **mc.to_dict()},
           "secondary_token_f1": _metrics_block(m2, v2)}
    with open(os.path.join(a.run_dir, "prereg_result.json"), "w") as f:
        json.dump(res, f, indent=2)

    def row(name, m, v=None):
        return (f"| {name} | {m.n} | {m.n_flagged} | {m.base_rate:.3f} | {m.precision:.3f} | "
                f"{m.recall:.3f} | {m.lift:.3f} | {m.permutation_p:.4f} | "
                f"{v.label if v else '—'} |")

    md = f"""# Pre-registered result — {os.path.basename(a.run_dir.rstrip('/'))}

**DECISION: {decision}** (rules: PREREGISTRATION_D8.md)

Judge validity: {agree}/{len(ann)} agreement with blind annotation = {agreement:.3f}
(threshold {AGREEMENT_MIN}) -> {"VALID" if judge_valid else "INVALID"}.
Judge labels: {judge_counts}

| analysis | n | flagged | base rate | precision | recall | lift | p | verdict |
|---|---|---|---|---|---|---|---|---|
{row("PRIMARY judge, all", m1, v1)}
{row(f"judge, >= {floor} tok (diagnostic)", mc)}
{row("secondary token-F1, all", m2, v2)}

Primary verdict reasons: {"; ".join(v1.reasons) or "none"}
"""
    with open(os.path.join(a.run_dir, "PREREG_RESULT.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
