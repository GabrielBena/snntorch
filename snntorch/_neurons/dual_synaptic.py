import torch
import torch.nn as nn
from .synaptic import Synaptic


class DualThresholdSynaptic(Synaptic):
    """"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "threshold_pos" not in kwargs:
            self.threshold_pos = self.threshold
        if "threshold_neg" not in kwargs:
            self.threshold_neg = -self.threshold

    def fire_graded(self, mem):
        """Generate spikes if mem > threshold+ or mem < threshold-
        Returns spk."""
        if self.state_quant:
            mem = self.state_quant(mem)

        mem_shift_pos = mem - self.threshold_pos
        mem_shift_neg = mem - self.threshold_neg

        spk_pos = self.spike_grad(mem_shift_pos)
        spk_neg = self.spike_grad(-mem_shift_neg)

        spk = spk_pos - spk_neg

        return spk

    def fire(self, mem):
        return self.fire_graded(mem)

    def mem_reset(self, mem, detach=True):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        if self.reset_mechanism == "subtract":
            mem_shift_pos = mem - self.threshold_pos
            mem_shift_neg = mem - self.threshold_neg

            reset_pos = self.spike_grad(mem_shift_pos).clone()
            reset_neg = self.spike_grad(-mem_shift_neg).clone()
            if detach:
                reset_pos = reset_pos.detach()
                reset_neg = reset_neg.detach()
            reset = reset_pos - reset_neg
        elif self.reset_mechanism == "zero":
            mem_shift_pos = mem - self.threshold_pos
            mem_shift_neg = mem - self.threshold_neg
            reset_pos = self.spike_grad(mem_shift_pos).clone()
            reset_neg = self.spike_grad(-mem_shift_neg).clone()
            if detach:
                reset_pos = reset_pos.detach()
                reset_neg = reset_neg.detach()
            reset = reset_pos + reset_neg
        else:
            reset = super().mem_reset(mem)

        return reset
