#!/usr/bin/env python3
"""
OBSERVER 2 audit of stage_localization (CPU only, read-only on run files).

  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe audits/observer2_analysis.py

Inputs: runs/llama3.2-3b-d8-full/{token_traces,judge_labels,items}.jsonl, mechanism_codes.tsv,
        audits/observer2_first_false_tokens.jsonl (labelled blind to entropy),
        audits/observer2_true_commit_tokens.jsonl (labelled blind to entropy),
        runs/qwen2.5-3b-d6d7/items.jsonl (independent model, for the difficulty test).
Output: audits/observer2_results.json (+ stdout tables).
"""
import sys, os, re, json, csv
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
D = os.path.join(ROOT, "runs", "llama3.2-3b-d8-full")
rng = np.random.default_rng(0)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def split_idx(tokens):
    """Same rule as scripts/stage_localization.split_headline; returns last headline token index."""
    text = "".join(tokens)
    m = re.search(r"[.!?](\s|$)", text)
    cut = m.end() if m else len(text)
    pos = 0
    for k, piece in enumerate(tokens):
        pos += len(piece)
        if pos >= cut:
            return k
    return len(tokens) - 1


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum()
                  + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def auc_ci(pos, neg, B=1000):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    s = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return [round(auc(pos, neg), 3), [round(float(x), 3) for x in np.nanpercentile(s, [2.5, 97.5])],
            int(len(pos)), int(len(neg))]


STOP = set("""a an the and or but if of to in on at by for with from as into onto over under about
between through during before after above below up down out off than then so such that this these those
it its it's is are was were be been being am do does did done have has had having can could may might
must shall should will would not no nor only own same too very just also often generally typically
usually however additionally furthermore therefore thus while which who whom whose what when where why
how all any both each few more most other some i you he she we they me him her us them my your his our
their there here because due one's s t""".split())


def is_eot(p):
    return p.startswith("<|") and p.endswith("|>")


def classify(tokens):
    """Per token: punct | function | content | number | cont (subword continuation) | eot.
    Also returns full word for word-initial tokens."""
    cls, words = [], []
    for k, p in enumerate(tokens):
        if is_eot(p):
            cls.append("eot"); words.append(None); continue
        s = p.strip()
        if not any(ch.isalnum() for ch in s):
            cls.append("punct"); words.append(None); continue
        prev = tokens[k - 1] if k else ""
        initial = k == 0 or p[:1].isspace() or (prev[-1:].isspace()) or (prev.strip() and not prev.strip()[-1].isalnum() and prev.strip()[-1] not in "'-" and p[:1].isalnum())
        if not initial:
            cls.append("cont"); words.append(None); continue
        # reconstruct full word
        w = s
        j = k + 1
        while j < len(tokens) and not is_eot(tokens[j]) and tokens[j][:1].isalnum() and not tokens[j][:1].isspace():
            w += tokens[j]; j += 1
        w = re.sub(r"[^a-z0-9']", "", w.lower())
        words.append(w)
        if w in STOP or w.rstrip("'s") in STOP:
            cls.append("function")
        elif w and w[0].isdigit():
            cls.append("number")
        else:
            cls.append("content")
    return cls, words


def main():
    from run_truthfulqa import load_items
    ev = load_items(817, seed=0)[60:]
    tr = {r["idx"]: r for r in jl(os.path.join(D, "token_traces.jsonl"))}
    judge = {r["idx"]: r["judge"] for r in jl(os.path.join(D, "judge_labels.jsonl"))}
    codes = {int(c["idx"]): c for c in csv.DictReader(open(os.path.join(D, "mechanism_codes.tsv"),
                                                           encoding="utf-8"), delimiter="\t")}
    ff = {r["idx"]: r for r in jl(os.path.join(ROOT, "audits", "observer2_first_false_tokens.jsonl"))}
    tc = {r["idx"]: r for r in jl(os.path.join(ROOT, "audits", "observer2_true_commit_tokens.jsonl"))}
    out = {}

    E = {i: np.asarray(t["entropy"], float) for i, t in tr.items()}
    T = {i: t["tokens"] for i, t in tr.items()}
    H = {i: split_idx(T[i]) for i in tr}
    refusal = {i: "".join(T[i]).strip().lower().startswith("i have no comment") for i in tr}
    true_all = sorted(i for i, j in judge.items() if j == "TRUE")
    true_sub = [i for i in true_all if not refusal[i]]
    real = sorted(i for i, c in codes.items() if c["code"] != "not_false")
    notfalse = sorted(i for i, c in codes.items() if c["code"] == "not_false")
    lure = sorted(i for i, c in codes.items() if c["note"].startswith("lure"))

    def body(i):  # drop trailing end-of-turn
        e = E[i]
        return e[:-1] if T[i] and is_eot(T[i][-1]) else e

    def h_mean(i): return float(E[i][:H[i] + 1].mean())
    def h_first3(i): return float(E[i][:min(3, H[i] + 1)].mean())
    def e_mean(i):
        e = body(i)[H[i] + 1:]
        return float(e.mean()) if len(e) >= 5 else np.nan

    # ---------------- 0. composition confound -----------------
    comp = {"n_true": len(true_all), "n_true_refusal": int(sum(refusal[i] for i in true_all)),
            "n_real_err": len(real), "n_real_err_refusal": int(sum(refusal[i] for i in real)),
            "n_notfalse_refusal": int(sum(refusal[i] for i in notfalse))}
    comp["auc_vs_true_all"] = {k: auc_ci([f(i) for i in real], [f(i) for i in true_all])
                               for k, f in [("h_first3", h_first3), ("h_mean", h_mean), ("e_mean", e_mean)]}
    comp["auc_vs_true_substantive"] = {k: auc_ci([f(i) for i in real], [f(i) for i in true_sub])
                                       for k, f in [("h_first3", h_first3), ("h_mean", h_mean), ("e_mean", e_mean)]}
    comp["lure_vs_true_substantive"] = {k: auc_ci([f(i) for i in lure], [f(i) for i in true_sub])
                                        for k, f in [("h_mean", h_mean), ("e_mean", e_mean)]}
    comp["notfalse_vs_true_substantive_h_mean"] = auc_ci([h_mean(i) for i in notfalse], [h_mean(i) for i in true_sub])
    comp["mean_headline_len"] = {"real": float(np.mean([H[i] + 1 for i in real])),
                                 "true_all": float(np.mean([H[i] + 1 for i in true_all])),
                                 "true_sub": float(np.mean([H[i] + 1 for i in true_sub]))}
    out["composition"] = comp

    # ---------------- 1. first-false-token analysis -----------------
    pos_ref = {}  # TRUE-substantive entropy by absolute position
    for i in true_sub:
        for k, v in enumerate(body(i)):
            pos_ref.setdefault(k, []).append(v)

    rows = []
    for i, r in ff.items():
        e = body(i); k = r["first_false_tok"]; h = H[i]
        others = np.delete(e, k)
        pre = e[:k]
        rk = float((e < e[k]).mean() + 0.5 * (e == e[k]).mean())  # within-answer percentile
        ref = np.asarray(pos_ref.get(k, []))
        rows.append({
            "idx": i, "group": r["group"], "ff": k, "headline_end": h, "n": len(e),
            "in_headline": k <= h, "conf": r["confidence"],
            "ent_ff": float(e[k]), "ent_prev": float(e[k - 1]) if k else np.nan,
            "within_pctile": rk, "is_argmax": int(np.argmax(e)) == k,
            "rank": int((e > e[k]).sum()) + 1,
            "ratio_to_median": float(e[k] / max(np.median(others), 1e-6)),
            "pre_mean": float(pre.mean()) if len(pre) else np.nan,
            "post_mean": float(e[k + 1:].mean()) if len(e) > k + 1 else np.nan,
            "pos_pctile_vs_true": float((ref < e[k]).mean()) if len(ref) else np.nan,
            "headline_minus_ff_mean": float(np.delete(e[:h + 1], k).mean()) if k <= h and h > 0 else np.nan,
        })
    # alternative (earlier) labels
    for row in rows:
        a = ff[row["idx"]]["first_false_tok_alt"]
        row["ff_alt"] = a
        row["alt_in_headline"] = (a <= row["headline_end"]) if a is not None else None
    out["ff_rows"] = rows

    def summ(rs):
        if not rs:
            return {}
        return {"n": len(rs),
                "in_headline": int(sum(r["in_headline"] for r in rs)),
                "ff_at_pos0": int(sum(r["ff"] == 0 for r in rs)),
                "median_within_pctile": float(np.median([r["within_pctile"] for r in rs])),
                "frac_top10pct_within": float(np.mean([r["within_pctile"] >= 0.9 for r in rs])),
                "frac_argmax": float(np.mean([r["is_argmax"] for r in rs])),
                "median_ent_ff": float(np.median([r["ent_ff"] for r in rs])),
                "median_ratio_to_median": float(np.median([r["ratio_to_median"] for r in rs])),
                "median_pos_pctile_vs_true": float(np.nanmedian([r["pos_pctile_vs_true"] for r in rs]))}
    groups = sorted(set(r["group"] for r in rows))
    out["ff_summary"] = {"ALL": summ(rows), **{g: summ([r for r in rows if r["group"] == g]) for g in groups},
                         "ff>0": summ([r for r in rows if r["ff"] > 0]),
                         "high_conf": summ([r for r in rows if r["conf"] == "high"])}

    # TRUE commitment tokens (blind-labelled) for comparison
    trows = []
    for i, r in tc.items():
        e = body(i); k = r["commit_tok"]
        trows.append({"idx": i, "refusal": r["refusal"], "ent": float(e[k]),
                      "within_pctile": float((e < e[k]).mean() + 0.5 * (e == e[k]).mean()),
                      "is_argmax": int(np.argmax(e)) == k})
    out["true_commit_rows"] = trows
    sub = [r for r in trows if not r["refusal"]]
    out["true_commit_summary"] = {
        "substantive": {"n": len(sub), "median_ent": float(np.median([r["ent"] for r in sub])),
                        "median_within_pctile": float(np.median([r["within_pctile"] for r in sub])),
                        "frac_argmax": float(np.mean([r["is_argmax"] for r in sub])),
                        "frac_top10pct_within": float(np.mean([r["within_pctile"] >= 0.9 for r in sub]))},
        "refusal": {"n": len(trows) - len(sub),
                    "median_ent": float(np.median([r["ent"] for r in trows if r["refusal"]]))},
        "auc_ent_ff_vs_true_substantive_commit": auc_ci([r["ent_ff"] for r in rows], [r["ent"] for r in sub], B=500),
        "auc_within_pctile_ff_vs_true_commit": auc_ci([r["within_pctile"] for r in rows],
                                                      [r["within_pctile"] for r in sub], B=500)}

    # Pre-commitment tokens: errors with ff>=2, tokens before ff, vs TRUE-substantive at the same positions
    pre_err, pre_ref = [], []
    for r in rows:
        if r["ff"] >= 2:
            e = body(r["idx"])
            for k in range(r["ff"]):
                ref = pos_ref.get(k, [])
                if ref:
                    pre_err.append(float((np.asarray(ref) < e[k]).mean()))
    out["pre_commit_position_percentile_vs_true"] = {
        "n_tokens": len(pre_err), "median": float(np.median(pre_err)) if pre_err else None,
        "mean": float(np.mean(pre_err)) if pre_err else None,
        "note": "0.5 = same as TRUE-substantive at that position; >0.5 = elevated BEFORE the false token"}

    # ---------------- 2. rationalization: explanation by token class -----------------
    def expl_class_means(i, words_q):
        tks = T[i]; cls, words = classify(tks)
        e = E[i]; h = H[i]
        seen = set(words_q)
        for k in range(h + 1):
            if words[k]: seen.add(words[k])
        acc = {}
        for k in range(h + 1, len(tks)):
            c = cls[k]
            if c == "eot":
                continue
            if c == "content":
                c2 = "content_copied" if words[k] in seen else "content_novel"
                acc.setdefault(c2, []).append(e[k])
            acc.setdefault(c, []).append(e[k])
            if words[k]:
                seen.add(words[k])
        return acc
    qwords = {i: set(re.sub(r"[^a-z0-9' ]", " ", ev[i]["question"].lower()).split()) for i in tr}
    C = {i: expl_class_means(i, qwords[i]) for i in tr}
    cls_names = ["punct", "function", "cont", "content", "content_copied", "content_novel", "number"]

    def cm(i, c, minn=3):
        v = C[i].get(c, [])
        return float(np.mean(v)) if len(v) >= minn else np.nan
    expl = {}
    for c in cls_names:
        expl[c] = {"real_vs_true_sub": auc_ci([cm(i, c) for i in real], [cm(i, c) for i in true_sub]),
                   "lure_vs_true_sub": auc_ci([cm(i, c) for i in lure], [cm(i, c) for i in true_sub]),
                   "mean_real": float(np.nanmean([cm(i, c) for i in real])),
                   "mean_true_sub": float(np.nanmean([cm(i, c) for i in true_sub]))}
    # class composition of explanation
    def frac(i, c):
        n = sum(len(C[i].get(x, [])) for x in ["punct", "function", "cont", "content", "number"])
        return len(C[i].get(c, [])) / n if n else np.nan
    expl["composition"] = {c: {"real": float(np.nanmean([frac(i, c) for i in real])),
                               "true_sub": float(np.nanmean([frac(i, c) for i in true_sub])),
                               "lure": float(np.nanmean([frac(i, c) for i in lure]))}
                           for c in ["punct", "function", "cont", "content", "number"]}
    expl["copied_share_of_content"] = {
        g: float(np.nanmean([len(C[i].get("content_copied", [])) / max(len(C[i].get("content", [])), 1)
                             for i in ids if C[i].get("content")]))
        for g, ids in [("real", real), ("true_sub", true_sub), ("lure", lure)]}

    # Lexically matched: same word type, word-initial content tokens in explanations
    by_type = {"real": {}, "true": {}}
    for g, ids in [("real", real), ("true", true_sub)]:
        for i in ids:
            tks = T[i]; cls, words = classify(tks); h = H[i]
            for k in range(h + 1, len(tks)):
                if cls[k] == "content":
                    by_type[g].setdefault(words[k], []).append(E[i][k])
    shared = [w for w in by_type["real"] if w in by_type["true"]
              and len(by_type["real"][w]) >= 3 and len(by_type["true"][w]) >= 3]
    diffs = np.array([np.mean(by_type["real"][w]) - np.mean(by_type["true"][w]) for w in shared])
    # bootstrap over word types
    bs = [rng.choice(diffs, len(diffs)).mean() for _ in range(2000)] if len(diffs) else [np.nan]
    expl["lexically_matched_content"] = {
        "n_types": len(shared), "mean_diff_real_minus_true": float(diffs.mean()) if len(diffs) else None,
        "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
        "frac_types_real_higher": float((diffs > 0).mean()) if len(diffs) else None}
    # position-matched: explanation tokens bucketed by offset from explanation start
    pm = {}
    for b0, b1 in [(0, 10), (10, 30), (30, 60), (60, 200)]:
        def m(i):
            e = body(i)[H[i] + 1:][b0:b1]
            return float(e.mean()) if len(e) >= 3 else np.nan
        pm[f"{b0}-{b1}"] = auc_ci([m(i) for i in real], [m(i) for i in true_sub], B=500)
    expl["position_bucket_auc_real_vs_true_sub"] = pm
    # within-answer: explanation relative to the answer's own headline
    rel = lambda i: e_mean(i) - h_mean(i)
    expl["expl_minus_headline_auc_real_vs_true_sub"] = auc_ci([rel(i) for i in real], [rel(i) for i in true_sub])
    out["explanation"] = expl

    # ---------------- 3. difficulty -----------------
    diff = {}
    # (a) real errors vs judge-FALSE-but-not-false: both flagged by judge, only one is wrong
    diff["real_vs_notfalse"] = {k: auc_ci([f(i) for i in real], [f(i) for i in notfalse])
                                for k, f in [("h_first3", h_first3), ("h_mean", h_mean), ("e_mean", e_mean)]}
    # (b) independent model's view of the question: Qwen2.5-3B (d6d7 run, same question order)
    qp = os.path.join(ROOT, "runs", "qwen2.5-3b-d6d7", "items.jsonl")
    if os.path.exists(qp):
        q = {r["idx"]: r for r in jl(qp)}
        aligned = [i for i in q if i in tr and q[i]["question"] == ev[i]["question"]]
        diff["qwen_aligned_n"] = len(aligned)
        qent = {i: q[i]["mean_entropy"] for i in aligned}
        qhal = {i: bool(q[i]["label"]["hallucinated"]) for i in aligned}
        lab = {i: 1 for i in real}; lab.update({i: 0 for i in true_sub})
        A = [i for i in aligned if i in lab]
        y = np.array([lab[i] for i in A]); x1 = np.array([h_mean(i) for i in A]); x2 = np.array([qent[i] for i in A])
        diff["qwen_entropy_predicts_llama_error_auc"] = auc_ci(x2[y == 1], x2[y == 0])
        diff["llama_hmean_predicts_llama_error_auc_same_subset"] = auc_ci(x1[y == 1], x1[y == 0])
        diff["corr_llama_hmean_qwen_entropy"] = float(np.corrcoef(x1, x2)[0, 1])
        # stratify by Qwen entropy tercile: within each, Llama headline AUC
        terc = np.quantile(x2, [1 / 3, 2 / 3]); strat = {}
        for name, lo, hi in [("low", -1, terc[0]), ("mid", terc[0], terc[1]), ("high", terc[1], 99)]:
            m = (x2 > lo) & (x2 <= hi)
            strat[name] = auc_ci(x1[m & (y == 1)], x1[m & (y == 0)], B=500)
        diff["llama_hmean_auc_within_qwen_entropy_tercile"] = strat
        # logistic regression (numpy IRLS), standardized predictors
        def logit(X, y):
            X = np.column_stack([np.ones(len(y)), X]); b = np.zeros(X.shape[1])
            for _ in range(50):
                p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
                b = b + np.linalg.solve(X.T @ (X * W[:, None]) + 1e-8 * np.eye(len(b)), X.T @ (y - p))
            p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]))))
            return b, se
        z = lambda v: (v - v.mean()) / v.std()
        b, se = logit(np.column_stack([z(x1), z(x2)]), y)
        diff["logit_llama_err_on_llama_hmean_and_qwen_entropy"] = {
            "coef": [float(v) for v in b], "se": [float(v) for v in se], "terms": ["const", "llama_h_mean", "qwen_mean_entropy"]}
        # qwen token-F1 'hallucinated' label (noisy) as question-hardness flag
        hq = np.array([qhal[i] for i in A])
        diff["llama_hmean_auc_by_qwen_label"] = {
            "qwen_wrong": auc_ci(x1[hq & (y == 1)], x1[hq & (y == 0)], B=500),
            "qwen_right": auc_ci(x1[~hq & (y == 1)], x1[~hq & (y == 0)], B=500)}
    # (c) the pre-commitment test is in out['pre_commit_position_percentile_vs_true']
    out["difficulty"] = diff

    json.dump(out, open(os.path.join(ROOT, "audits", "observer2_results.json"), "w"), indent=1, default=float)

    # ---------------- print -----------------
    P = lambda *a: print(*a)
    P("COMPOSITION", json.dumps(comp, indent=1))
    P("\nFIRST-FALSE TOKENS by group"); [P(f"  {g:16s}", json.dumps(v)) for g, v in out["ff_summary"].items()]
    P("\nTRUE commit", json.dumps(out["true_commit_summary"], indent=1))
    P("\nPRE-COMMIT", out["pre_commit_position_percentile_vs_true"])
    P("\nPER-ROW")
    for r in sorted(rows, key=lambda r: (r["group"], r["idx"])):
        P(f"  {r['idx']:4d} {r['group']:15s} ff={r['ff']:3d} h_end={r['headline_end']:3d} inH={int(r['in_headline'])} "
          f"ent={r['ent_ff']:.2f} prev={r['ent_prev']:.2f} pct={r['within_pctile']:.2f} rank={r['rank']:3d}/{r['n']} "
          f"posPct={r['pos_pctile_vs_true']:.2f} pre={r['pre_mean']:.2f} post={r['post_mean']:.2f} {r['conf']}")
    P("\nEXPLANATION", json.dumps(expl, indent=1))
    P("\nDIFFICULTY", json.dumps(diff, indent=1))


if __name__ == "__main__":
    main()
