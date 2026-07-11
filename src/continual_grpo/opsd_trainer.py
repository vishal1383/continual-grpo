"""GRPOTrainer subclass implementing C-OPSD (proposal Aim 2, Eq. 8) on TRL 0.15.

For `method` "copsd" or "combined" the scalar-reward GRPO objective is
replaced by contrastive on-policy self-distillation: within each generation
group, verified-correct completions become positive distillation targets
(reverse KL to the reference at the tokens after the traces diverge) and
verified-incorrect ones enter a bounded log-likelihood margin.  Batches whose
groups contain no correct/incorrect pair fall back to the plain GRPO loss so
training never stalls.  For "grpo"/"skill_ortho" this class defers entirely
to the parent.
"""
from __future__ import annotations

import torch
from trl import GRPOTrainer
from trl.trainer.utils import selective_log_softmax

from .contrastive_opsd import contrastive_opsd_loss


class RecordingReward:
    """Wraps the correctness verifier so per-completion labels reach the loss."""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.last: list[float] = []

    def __call__(self, **kwargs):
        self.last = list(self.fn(**kwargs))
        return self.last


def divergence_masks(pos_ids: torch.Tensor, neg_ids: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """Mark positive tokens from the first index where the paired traces differ."""
    masks = torch.zeros_like(pos_mask)
    for row, (pos, neg) in enumerate(zip(pos_ids, neg_ids)):
        different = (pos != neg).nonzero()
        start = int(different[0]) if different.numel() else 0
        masks[row, start:] = 1
    return (masks * pos_mask).float()


def select_pairs(correct: list[float], group_size: int) -> tuple[list[int], list[int]]:
    """Pair every verified-correct completion with a wrong one from its group."""
    positives, negatives = [], []
    for group_start in range(0, len(correct), group_size):
        group = correct[group_start:group_start + group_size]
        wrong = [group_start + i for i, c in enumerate(group) if c <= 0.5]
        if not wrong or all(c <= 0.5 for c in group):
            continue
        for i, c in enumerate(group):
            if c > 0.5:
                positives.append(group_start + i)
                negatives.append(wrong[0])
    return positives, negatives


class OPSDTrainer(GRPOTrainer):
    def __init__(self, *args, method: str = "grpo", opsd_margin: float = 0.2,
                 opsd_negative_weight: float = 0.1, opsd_temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.opsd_margin = opsd_margin
        self.opsd_negative_weight = opsd_negative_weight
        self.opsd_temperature = opsd_temperature
        self._correctness = next(f for f in self.reward_funcs if isinstance(f, RecordingReward))

    def _completion_logits(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       logits_to_keep=logits_to_keep + 1).logits
        return logits[:, :-1, :][:, -logits_to_keep:]

    def _reference_logits(self, input_ids, attention_mask, logits_to_keep):
        with torch.no_grad():
            if self.ref_model is not None:
                return self._completion_logits(self.ref_model, input_ids, attention_mask, logits_to_keep)
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                return self._completion_logits(self.model, input_ids, attention_mask, logits_to_keep)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.method not in ("copsd", "combined"):
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        logits = self._completion_logits(model, input_ids, attention_mask, logits_to_keep)
        per_token_logps = selective_log_softmax(logits, completion_ids)

        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps)
                        - (ref_per_token_logps - per_token_logps) - 1)
        lengths = completion_mask.sum(dim=1).clamp_min(1)
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / lengths).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        positives, negatives = select_pairs(self._correctness.last, self.num_generations)
        self._metrics["opsd_pairs"].append(float(len(positives)))
        if not positives:
            advantages = inputs["advantages"]
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            return ((per_token_loss * completion_mask).sum(dim=1) / lengths).mean()

        pos = torch.tensor(positives, device=logits.device)
        neg = torch.tensor(negatives, device=logits.device)
        ref_logits = self._reference_logits(input_ids[pos], attention_mask[pos], logits_to_keep)
        diverge = divergence_masks(completion_ids[pos], completion_ids[neg], completion_mask[pos])
        loss, stats = contrastive_opsd_loss(
            logits[pos].float(), ref_logits.float(),
            per_token_logps[pos], per_token_logps[neg],
            completion_mask[pos].float(), completion_mask[neg].float(), diverge,
            margin=self.opsd_margin, negative_weight=self.opsd_negative_weight,
            temperature=self.opsd_temperature)
        for name, value in stats.items():
            self._metrics[name].append(float(value))
        return loss
