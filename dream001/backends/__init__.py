"""
Entropy sources. Every backend yields (text, entropies_nats, coverage).

coverage = probability mass actually observed. 1.0 for full-vocab backends;
< 1.0 for the top-k API tier, where entropy is systematically UNDERestimated.
The bias is reported per token so it is visible, never silent.
"""
from .synthetic import SyntheticBackend, SyntheticTrace  # noqa: F401

__all__ = ["SyntheticBackend", "SyntheticTrace", "get_backend"]


def get_backend(kind: str, **kw):
    if kind == "synthetic":
        return SyntheticBackend(**kw)
    if kind == "hf":
        from .hf import HFBackend
        return HFBackend(**kw)
    if kind == "api":
        from .openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(**kw)
    raise ValueError(f"unknown backend {kind!r}")
