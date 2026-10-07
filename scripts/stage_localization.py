#!/usr/bin/env python3
"""
EXPLORATORY. Headline-vs-explanation entropy by hallucination mechanism, with
difficulty controls. Reads token_traces.jsonl (scripts/token_traces.py),
judge_labels.jsonl and mechanism_codes.tsv; writes stage_localization.json.

  python scripts/stage_localization.py runs/<run> --n 757 --n-calibration 60

Headline = generated tokens up to and including the first sentence end
([.!?] followed by whitespace or end of text). Explanation = the rest, used
only when >= 5 tokens. AUCs compare each group against judge-TRUE answers.
This measures WHERE uncertainty is elevated; it does not locate the first false
token or establish a cause.
"""
import sys, os, re, json, csv, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))


def split_headline(tokens, ent):
    ent = np.asarray(ent, dtype=float)
    text = "".join(tokens)
    m = re.search(r"[.!?](\s|$)", text)
    cut = m.end() if m else len(text)
    pos = 0
    for k, piece in enumerate(tokens):
        pos += len(piece)
        if pos >= cut:
            return ent[:k + 1], ent[k + 1:]
    return ent, ent[:0]


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum()
                  + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def boot_ci(pos, neg, rng, B=1000):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    s = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return [float(x) for x in np.nanpercentile(s, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--n-calibration", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = a.run_dir

    from run_truthfulqa import load_items
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    topic = {r["question"]: r["category"] for r in ds}
    ev = load_items(a.n + a.n_calibration, seed=a.seed)[a.n_calibration:]

    tr = {}
    for l in open(os.path.join(d, "token_traces.jsonl"), encoding="utf-8"):
        r = json.loads(l); tr[r["idx"]] = r
    judge = {}
    for l in open(os.path.join(d, "judge_labels.jsonl"), encoding="utf-8"):
        r = json.loads(l); judge[r["idx"]] = r["judge"]
    codes = {int(c["idx"]): c for c in csv.DictReader(
        open(os.path.join(d, "mechanism_codes.tsv"), encoding="utf-8"), delimiter="\t")}

    feat = {}
    for i, t in tr.items():
        h, e = split_headline(t["tokens"], t["entropy"])
        feat[i] = {"h_mean": h.mean() if len(h) else np.nan,
                   "h_first3": h[:3].mean() if len(h) else np.nan,
                   "e_mean": e.mean() if len(e) >= 5 else np.nan,
                   "h_len": len(h)}

    true = sorted(i for i, j in judge.items() if j == "TRUE")
    real = sorted(i for i, c in codes.items() if c["code"] != "not_false")
    groups = {}
    for i, c in codes.items():
        groups.setdefault(c["code"], []).append(i)
    groups["famous_name_lure"] = [i for i, c in codes.items() if c["note"].startswith("lure")]
    groups["reasoning_excl_lure"] = [i for i, c in codes.items()
                                     if c["code"] == "reasoning" and not c["note"].startswith("lure")]
    groups["ALL_real_errors"] = real

    rng = np.random.default_rng(0)
    out = {"by_mechanism": {}, "controls": {}}
    order = ["fabrication", "fiction_as_fact", "misconception", "reasoning_excl_lure",
             "false_premise", "overclaim", "famous_name_lure", "not_false", "ALL_real_errors"]
    print(f"{'group':22s} {'n':>3s}  {'headline first-3':>20s}  {'headline mean':>20s}  {'explanation':>20s}")
    for k in order:
        ids = groups.get(k, [])
        if not ids:
            continue
        row = {"n": len(ids)}
        line = f"{k:22s} {len(ids):3d}"
        for key in ("h_first3", "h_mean", "e_mean"):
            pos = [feat[i][key] for i in ids]; neg = [feat[i][key] for i in true]
            a_ = auc(pos, neg); lo, hi = boot_ci(pos, neg, rng)
            row[key] = {"auc": a_, "ci95": [lo, hi]}
            line += f"  {a_:.2f} [{lo:.2f},{hi:.2f}]    "
        out["by_mechanism"][k] = row
        print(line)

    # Control 1: answers the judge called FALSE that review found not false.
    nf = groups.get("not_false", [])
    out["controls"]["not_false_vs_true_headline"] = auc([feat[i]["h_mean"] for i in nf],
                                                        [feat[i]["h_mean"] for i in true])
    # Control 2: within original TruthfulQA topic (fixed before generation).
    tot = w = 0.0
    for c in sorted(set(topic.values())):
        p = [feat[i]["h_mean"] for i in real if topic[ev[i]["question"]] == c]
        n = [feat[i]["h_mean"] for i in true if topic[ev[i]["question"]] == c]
        if len(p) >= 3 and len(n) >= 3:
            tot += auc(p, n) * len(p); w += len(p)
    out["controls"]["within_topic_headline_auc"] = tot / w if w else float("nan")
    out["controls"]["within_topic_errors_covered"] = int(w)
    # Control 3: riddle-format (Confusion:*) questions only.
    rq = [i for i in range(len(ev)) if topic[ev[i]["question"]].startswith("Confusion")]
    rt = [i for i in rq if i in set(true)]; rf = [i for i in rq if i in set(real)]
    out["controls"]["riddle_only"] = {"n_true": len(rt), "n_err": len(rf),
                                      "headline_auc": auc([feat[i]["h_mean"] for i in rf],
                                                          [feat[i]["h_mean"] for i in rt])}
    print("\ncontrols:", json.dumps(out["controls"], indent=1))
    with open(os.path.join(d, "stage_localization.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
