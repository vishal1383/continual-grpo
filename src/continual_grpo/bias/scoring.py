"""Length-normalized continuation log-probability scoring.

``continuation_logprobs`` is the shared likelihood scorer behind CrowS-Pairs,
StereoSet, WinoBias, and UnStereoEval: for each (context, continuation) pair it
returns the mean per-token log-probability of the continuation tokens under the
model. Copied unchanged from the original run_bias_evals.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def continuation_logprobs(model, tok, contexts: list[str], continuations: list[str], batch_size: int, max_length: int):
    if not contexts:
        return []
    scores: list[float] = []
    device = next(model.parameters()).device
    for start in range(0, len(contexts), batch_size):
        ctx_batch = contexts[start : start + batch_size]
        cont_batch = continuations[start : start + batch_size]
        texts = [c + t for c, t in zip(ctx_batch, cont_batch)]
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(device)
        out = model(**enc)
        logits = out.logits[:, :-1, :]
        target = enc["input_ids"][:, 1:]
        neg_logp = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
        ).view_as(target)
        token_lp = -neg_logp
        attn = enc["attention_mask"][:, 1:].bool()
        target_unpadded_pos = enc["attention_mask"].cumsum(dim=1)[:, 1:]
        ctx_lens = [
            len(tok(c, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"])
            for c in ctx_batch
        ]
        ctx_lens_t = torch.tensor(ctx_lens, device=device).unsqueeze(1)
        score_mask = attn & (target_unpadded_pos > ctx_lens_t)
        # If truncation removed the continuation, fall back to every non-pad token
        # after the first token so the row still gets a deterministic score.
        empty = score_mask.sum(dim=1) == 0
        if empty.any():
            score_mask[empty] = attn[empty]
        sums = (token_lp * score_mask.float()).sum(dim=1)
        lens = score_mask.sum(dim=1).clamp_min(1)
        scores.extend((sums / lens).detach().float().cpu().tolist())
        del enc, out, logits, target, neg_logp, token_lp, attn, target_unpadded_pos, ctx_lens_t, score_mask, sums, lens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores
