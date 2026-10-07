#!/usr/bin/env python3
"""
The hypothesis test. Requires GPU (hf) or a logprobs-capable endpoint (api).

  python scripts/run_truthfulqa.py --model meta-llama/Llama-3.2-3B-Instruct
  python scripts/run_truthfulqa.py --api http://localhost:8000/v1 --model <name>

Model choice: 3B at fp16 over 8B at 4-bit. Quantization distorts the logit
distribution that IS the signal.
"""
import sys, os, json, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dream001 import RegimeConfig
from dream001.harness import PROMPTS, build_reference, run_eval, write_outputs
from dream001.report import compute_metrics


def load_items(n, seed=0):
    from datasets import load_dataset
    # datasets>=3 requires the namespaced repo id; the bare alias no longer
    # resolves. Fall back so older pinned envs still load.
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    except Exception:
        ds = load_dataset("truthful_qa", "generation")["validation"]
    out = []
    for r in ds:
        out.append({"question": r["question"],
                    "correct_answers": r["correct_answers"],
                    "incorrect_answers": r["incorrect_answers"]})
    # TruthfulQA is category-grouped; an unshuffled first-N calibration
    # slice skews the D1 reference distribution (CHANGELOG 0.2.0).
    random.Random(seed).shuffle(out)
    return out[:n]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api", default=None, help="OpenAI-compatible base URL (DREAM-006 tier)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-calibration", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="calibration/eval split shuffle seed; recorded in RUN_SPEC")
    ap.add_argument("--outdir", default="runs/latest")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="concise",
                    help="concise = spec PROMPT; longform = D8 answer + short "
                         "explanation (declared deviation, SPEC_DEVIATIONS.md D8)")
    ap.add_argument("--require-no-evidence", action="store_true",
                    help="sec.7.2 SECONDARY ARM -- REFUSED in this build: the "
                         "two-term gate needs a per-token evidence_mask and no "
                         "backend emits one (SPEC_DEVIATIONS.md E4)")
    a = ap.parse_args()

    if a.require_no_evidence:
        # The gate has nothing to gate on: both backends return
        # (text, entropies, coverage) and carry no evidence signal. Refuse here,
        # before the model is even loaded, rather than write "secondary arm: ON"
        # into RUN_SPEC.md over single-term results (SPEC_DEVIATIONS.md E4).
        ap.error(
            "--require-no-evidence is not runnable in this build. The sec.7.2 "
            "two-term arm requires a per-token evidence_mask; no backend "
            "produces one, so the flag would score the single-term detector "
            "and label it two-term. Defining evidence_arrival operationally "
            "comes first, and it must not be tuned on synthetic data (D4). "
            "Re-run without the flag to test the single-term hypothesis; see "
            "SPEC_DEVIATIONS.md E4."
        )

    from dream001.backends import get_backend
    backend = (get_backend("api", base_url=a.api, model=a.model, temperature=a.temperature)
               if a.api else
               get_backend("hf", model=a.model, temperature=a.temperature))

    items = load_items(a.n + a.n_calibration, seed=a.seed)
    calib, evalset = items[:a.n_calibration], items[a.n_calibration:]

    cfg = RegimeConfig.short_sequence()

    print(f"calibration pass: {len(calib)} items (excluded from metrics)")
    prompt = PROMPTS[a.prompt]
    # D1 reference and eval MUST share a template (see build_reference).
    ref = build_reference(backend, [c["question"] for c in calib],
                          max_items=a.n_calibration, prompt=prompt)
    print(f"reference distribution: {ref.size} tokens")

    print(f"eval pass: {len(evalset)} items")
    records, m, v = run_eval(backend, evalset, config=cfg, reference=ref, prompt=prompt)
    write_outputs(records, m, v, cfg, outdir=a.outdir)

    # Length-confound diagnostic (FINDINGS.md). Lift is computed against the
    # full-set base rate, but no event can exist below the floor, and longer
    # answers are likelier to be hallucinations -- so full-set lift can be a
    # length artefact. Re-scored on the population the detector can act on.
    # DIAGNOSTIC ONLY: the verdict above is unchanged and stays ex-ante.
    floor = cfg.band_min_run + cfg.min_hold_run + 2   # + min_event_tokens
    cap = [r for r in records if r.n_tokens >= floor]
    cm = (compute_metrics([r.flagged for r in cap],
                          [r.label["hallucinated"] for r in cap])
          if cap else None)
    max_new = getattr(backend, "max_new_tokens", None)
    n_cap_hit = sum(r.n_tokens >= max_new for r in records) if max_new else 0
    lens = sorted(r.n_tokens for r in records)
    diag = {"event_floor_tokens": floor, "n_capable": len(cap),
            "n_hit_token_cap": n_cap_hit,
            "median_answer_tokens": lens[len(lens) // 2],
            "capable_metrics": cm.to_dict() if cm else None}
    with open(os.path.join(a.outdir, "length_diagnostic.json"), "w") as f:
        json.dump(diag, f, indent=2)
    cap_line = (f"precision {cm.precision:.4f} | base rate {cm.base_rate:.4f} | "
                f"lift {cm.lift:.4f} | p {cm.permutation_p:.4f} | "
                f"flagged {cm.n_flagged}/{cm.n}" if cm else "no capable answers")
    print(f"\nlength diagnostic: {len(cap)}/{len(records)} answers >= {floor} tok; "
          f"median {diag['median_answer_tokens']} tok; {n_cap_hit} hit the token cap")
    print(f"  among capable: {cap_line}")

    with open(os.path.join(a.outdir, "RUN_SPEC.md"), "w") as f:
        f.write(f"""# RUN_SPEC

- model: `{a.model}`
- backend: {"api (DREAM-006 tier, top-k)" if a.api else "hf full-vocab"}
- temperature: {a.temperature} ({"greedy" if a.temperature == 0 else "sampled"})
- seed: {a.seed} (calibration/eval shuffle)
- eval items: {len(evalset)} | calibration items: {len(calib)} (excluded)
- preset: short_sequence (D1 calibrated, D4 min_rise disabled)
- prompt: {a.prompt}{" (D8 declared deviation -- long-form elicitation)" if a.prompt == "longform" else " (spec)"}
- secondary arm (sec.7.2 two-term): off (not runnable -- no evidence source; SPEC_DEVIATIONS.md E4)
- label method: token_f1_v0 (noisy both directions; dilutes toward base rate)

## Result
precision {m.precision:.4f} | recall {m.recall:.4f} | lift {m.lift:.4f}
base rate {m.base_rate:.4f} | firing rate {m.firing_rate:.4f} | p {m.permutation_p:.4f}

**VERDICT: {v.label}**

## Length diagnostic (not a gate)
{len(cap)}/{len(records)} answers >= {floor}-token event floor; median answer
{diag["median_answer_tokens"]} tokens; {n_cap_hit} hit the generation cap.
Among capable answers: {cap_line}
""")
    print(f"\nwrote {a.outdir}/items.jsonl, truthfulqa_report.json, RUN_SPEC.md")
    sys.exit(0 if v.passed else 1)
