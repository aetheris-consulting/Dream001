"""
HuggingFace full-vocab backend.

CRITICAL: entropy is computed INSIDE the step and only the scalar is retained.
Materializing full-vocabulary logits across a 300-item run exhausts VRAM and
disk (ISCOT v1.2 sec.6.1). Never store the logit tensor.

Model choice per spec: Llama 3.2 3B at fp16, NOT 8B at 4-bit. Quantization
distorts the logit distribution that IS the signal; a quantized precision
number is uninterpretable.

D6 (SPEC_DEVIATIONS.md): the harness PROMPT is delivered through the model's
chat template when it has one, and generation stops at a turn boundary. Feeding
a completion-style prompt raw to an Instruct model produced base-model
continuation -- the model emitted its answer and then confabulated a fresh
system prompt, echoed the instruction text back, and ran to the token cap. The
entropy trace was then measured over that continuation rather than over an
answer. This is a declared deviation, not a silent fix.
"""

from __future__ import annotations
from typing import List, Tuple, Sequence, Optional

# Turn boundaries the completion-style PROMPT invites the model to continue past.
DEFAULT_STOP_STRINGS = ("\nQ:", "\nQuestion:", "\nA:")


class HFBackend:
    def __init__(self, model: str = "meta-llama/Llama-3.2-3B-Instruct",
                 device: str = "cuda", dtype: str = "float16",
                 max_new_tokens: int = 128, temperature: float = 0.0,
                 use_chat_template: bool = True,
                 stop_strings: Optional[Sequence[str]] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.stop_strings = tuple(DEFAULT_STOP_STRINGS if stop_strings is None
                                  else stop_strings)
        self.tok = AutoTokenizer.from_pretrained(model)
        try:                                    # transformers >=5 renamed the kwarg
            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype=getattr(torch, dtype), device_map=device)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()
        self.name = model
        self.use_chat_template = bool(
            use_chat_template and getattr(self.tok, "chat_template", None))

    def _entropy(self, logits) -> float:
        torch = self.torch
        x = logits.float()
        if self.temperature and self.temperature > 0:
            x = x / self.temperature
        logp = torch.log_softmax(x, dim=-1)
        return float(-(logp.exp() * logp).sum().item())  # nats; scalar only

    def _render(self, prompt: str) -> str:
        """D6: route through the chat template when the model has one."""
        if not self.use_chat_template:
            return prompt
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)

    def _truncate_at_stop(self, gen_ids, entropies: List[float]):
        """
        Cut text AND entropies at the same token, so the trace never covers
        tokens that are not part of the answer. Returns (text, entropies).
        """
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        hits = [text.find(s) for s in self.stop_strings if s in text]
        if not hits:
            return text, entropies
        cut_char = min(h for h in hits if h >= 0)

        # first generated token whose decoded prefix reaches into the stop string
        cut_tok = len(gen_ids)
        for k in range(1, len(gen_ids) + 1):
            if len(self.tok.decode(gen_ids[:k], skip_special_tokens=True)) > cut_char:
                cut_tok = k - 1
                break
        return text[:cut_char], entropies[:cut_tok]

    def generate(self, prompt: str) -> Tuple[str, List[float], List[float]]:
        """Greedy decoding is the reproducible default (README limitation 7)."""
        enc = self.tok(self._render(prompt), return_tensors="pt").to(self.model.device)
        kw = {}
        if self.stop_strings:
            kw = {"stop_strings": list(self.stop_strings), "tokenizer": self.tok}
        with self.torch.no_grad():
            try:
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    return_dict_in_generate=True, output_scores=True,
                    pad_token_id=self.tok.eos_token_id, **kw)
            except (TypeError, ValueError):      # older transformers: no stop_strings
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    return_dict_in_generate=True, output_scores=True,
                    pad_token_id=self.tok.eos_token_id)

        entropies = [self._entropy(s[0]) for s in out.scores]
        gen = out.sequences[0][enc["input_ids"].shape[1]:]
        n = min(len(entropies), len(gen))        # keep text and trace aligned
        gen, entropies = gen[:n], entropies[:n]
        text, entropies = self._truncate_at_stop(gen, entropies)
        return text, entropies, [1.0] * len(entropies)
