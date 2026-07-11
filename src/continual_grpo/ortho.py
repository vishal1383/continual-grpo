"""Aim 1 integration: skill-isolating gradient transform before optimizer.step().

The callback fires once per optimizer step (after backward and gradient
accumulation, before clipping), replacing every 2-D gradient with its
projected, spectrally restricted form from `spectral_update`.
"""
from __future__ import annotations

from transformers import TrainerCallback

from .spectral_update import ProtectedAxes, transform_gradients


class SkillOrthoCallback(TrainerCallback):
    def __init__(self, axes_by_parameter: dict[str, ProtectedAxes], target_rank: int = 1):
        self.axes_by_parameter = axes_by_parameter
        self.target_rank = target_rank

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is not None:
            transform_gradients(model, self.axes_by_parameter, self.target_rank)
