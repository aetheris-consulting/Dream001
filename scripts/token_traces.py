#!/usr/bin/env python3
"""
Recover per-token entropy traces for a finished run (run_eval stores only the
per-answer mean).

  python scripts/token_traces.py runs/<run> --model meta-llama/Llama-3.2-3B-Instruct --prompt longform

Teacher-forced: one forward pass over chat-rendered prompt + saved generation,
entropy read at each generated position. Same full-vocab, fp16 computation as
HFBackend, so it should reproduce the generation-time trace up to re-tokenization
and kernel noise. Each record carries its own check: token count and mean
entropy against the stored values. Writes token_traces.jsonl.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dream001.harness import PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="cuda")
    model.eval()

    out = open(os.path.join(a.run_dir, "token_traces.jsonl"), "w", encoding="utf-8")
    for line in open(os.path.join(a.run_dir, "items.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": PROMPTS[a.prompt].format(q=r["question"])}],
            tokenize=False, add_generation_prompt=True)
        # Tokenize the prompt EXACTLY as HFBackend.generate did: default
        # add_special_tokens, which prepends a second BOS to the chat template's
        # own (D10). The trace must reproduce the run's conditioning, defects
        # included, or it measures a different input.
        p_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
        g_ids = tok(r["generation"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # The run's trace includes the step that emitted end-of-turn; the saved
        # text drops it. Restore it when the counts say the answer ended on EOT.
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        if r["n_tokens"] == len(g_ids) + 1:
            g_ids = torch.cat([g_ids, torch.tensor([eot])])
        ids = torch.cat([p_ids, g_ids]).unsqueeze(0).to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        # the distribution that produced generated token k sits at position len(p)-1+k
        lp = torch.log_softmax(logits[len(p_ids) - 1: len(p_ids) - 1 + len(g_ids)], dim=-1)
        ent = (-(lp.exp() * lp).sum(-1)).tolist()
        pieces = [tok.decode([t]) for t in g_ids.tolist()]
        out.write(json.dumps({
            "idx": r["idx"], "tokens": pieces, "entropy": ent,
            "n_stored": r["n_tokens"], "n_recovered": len(ent),
            "mean_stored": r["mean_entropy"],
            "mean_recovered": sum(ent) / len(ent) if ent else 0.0}) + "\n")
    out.close()
    print("wrote token_traces.jsonl")


if __name__ == "__main__":
    main()
