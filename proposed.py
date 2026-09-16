"""Single-file gated steering and one-shot remasking. Requires PyTorch only.

Create one Proposed instance per response, with an eval-mode model and fitted
gate/CSD tensors. Model loading, tokenization and token sampling stay with the
caller. Only PyTorch and the Python standard library are imported.

Contract:
    state: torch.long [1, sequence], modified in place only by before_step.
    region: torch.bool with the same shape/device; the ORIGINAL answer slots.
    Model blocks return hidden [1, sequence, hidden_size], optionally as the
    first item of a tuple. model(state) must call the supplied blocks. Blocks
    are found at model.blocks, model.model.transformer.blocks (LLaDA) or
    model.model.layers (Dream); pass mask_id for models other than LLaDA.
    Gate vectors point toward risk. Steering vectors point toward REFUSAL.
    A steering vector is [hidden_size], or [4, hidden_size] for mask ratios
    (0.3, 0.5, 0.7, 0.9)..

Integration (model and checkpoints are supplied by the caller):
    policy = Proposed.from_llada(model, gate_checkpoint, csd_checkpoint, threshold=6.0)
    masks, remaining, commit_count = policy.before_step(
        state, region, steps_remaining=steps_left, commit_count=commit_count)
    output = policy.forward(state, region)
    # The caller samples output.logits and commits tokens at masks using its
    # decoding schedule. Repeat once per generation step until no masks remain.

before_step selects at most one remasking action per response. Once risk is
detected, steering remains active through completion. An initially armed policy
waits for committed tokens and sufficient budget before remasking. With
initial_only=True, an unarmed initial gate ends monitoring immediately.
"""
from abc import ABC, abstractmethod
import math

import torch

__all__ = ['Defense', 'Proposed']


def _blocks(model):
    """Transformer block list of a LLaDA or Dream model (or a wrapper exposing .blocks)."""
    if hasattr(model, 'blocks'):
        return model.blocks
    inner = getattr(model, 'model', None)
    if hasattr(inner, 'transformer'):
        return inner.transformer.blocks
    if hasattr(inner, 'layers'):
        return inner.layers
    raise AttributeError('cannot locate transformer blocks on the model')


class Defense(ABC):
    """One response's intervention lifecycle; the caller owns token sampling."""

    @abstractmethod
    def before_step(self, state, region, *, steps_remaining, commit_count):
        """Return updated (masks, remaining_count, commit_count)."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, state, region):
        """Run one generation forward, cleaning up temporary hooks afterward."""
        raise NotImplementedError


class Proposed(Defense):
    def __init__(self, model, *, gate_layer, gate_vector, gate_center, gate_scale,
                 threshold, steering_sites, mask_id=126336, strength=0.4,
                 max_remask_tokens=16, max_parallel_commit=2, initial_only=True,
                 mode='repair'):
        """steering_sites: sequence of (model block, refusal-direction tensor).

        Modes: baseline = initial gate + steering; steer = continuous detection
        without remasking; repair = detection + one-shot remasking. initial_only
        controls repair mode only. The steering transform uses additive CSD.
        """
        if model.training:
            raise ValueError('Call model.eval() before constructing the defense')
        if mode not in ('baseline', 'steer', 'repair'):
            raise ValueError('mode must be baseline, steer or repair')
        if not math.isfinite(gate_scale) or gate_scale <= 0:
            raise ValueError('gate_scale must be finite and positive')
        if not math.isfinite(threshold) or not math.isfinite(strength) or strength < 0:
            raise ValueError('threshold must be finite and strength nonnegative')
        if max_remask_tokens <= 0 or max_parallel_commit <= 0:
            raise ValueError('Token budgets must be positive')
        if gate_vector.ndim != 1 or gate_center.shape != gate_vector.shape:
            raise ValueError('Gate center and vector must be aligned 1D tensors')
        if not torch.isfinite(gate_center).all() or not torch.isfinite(gate_vector).all():
            raise ValueError('Gate tensors must be finite')
        if gate_vector.norm() == 0:
            raise ValueError('Gate vector must be nonzero')
        self.sites = list(steering_sites)
        if not self.sites:
            raise ValueError('At least one steering site is required')
        for block, vector in self.sites:
            if vector.ndim not in (1, 2) or (vector.ndim == 2 and vector.shape[0] != 4):
                raise ValueError('Directions must have shape [hidden] or [4, hidden]')
            if not torch.isfinite(vector).all() or (vector.norm(dim=-1) == 0).any():
                raise ValueError('Directions must be finite and nonzero')
        self.model, self.gate_layer = model, gate_layer
        self.gate_vector, self.gate_center = gate_vector, gate_center
        self.gate_scale, self.threshold = gate_scale, threshold
        self.mask_id, self.strength = mask_id, strength
        self.max_remask_tokens, self.max_parallel_commit = max_remask_tokens, max_parallel_commit
        self.initial_only, self.mode = initial_only, mode
        self.armed = self.done = False
        self.step = 0

    @classmethod
    def from_llada(cls, model, gate, csd, layers=(12, 16, 20, 24), **options):
        """Connect a loaded LLaDA model and tensor checkpoints (no downloads).

        gate: layer (1-based), vector, center, scale, threshold.
        csd: layers (1-based), vector [layers, hidden]. Optional mask_ratios and
        directions_by_ratio [4, layers, hidden] enable conditioned steering.
        Use torch.load(path, weights_only=True) to load these dictionaries.
        """
        blocks = _blocks(model)
        indices = [list(csd['layers']).index(layer) for layer in layers]
        if 'directions_by_ratio' in csd:
            if tuple(csd['mask_ratios']) != (0.3, 0.5, 0.7, 0.9):
                raise ValueError('Expected mask ratios (0.3, 0.5, 0.7, 0.9)')
            sites = [(blocks[layer - 1], csd['directions_by_ratio'][:, index])
                     for layer, index in zip(layers, indices)]
        else:
            sites = [(blocks[layer - 1], csd['vector'][index])
                     for layer, index in zip(layers, indices)]
        options.setdefault('threshold', gate['threshold'])
        return cls(model, gate_layer=blocks[gate['layer'] - 1],
                   gate_vector=gate['vector'], gate_center=gate['center'],
                   gate_scale=gate['scale'], steering_sites=sites, **options)

    @staticmethod
    def _validate(state, region):
        if (state.ndim != 2 or state.shape[0] != 1 or state.dtype != torch.long
                or region.shape != state.shape or region.dtype != torch.bool
                or region.device != state.device or not region.any()):
            raise ValueError('Expected long state and nonempty bool region shaped [1, sequence]')

    @torch.no_grad()
    def score(self, state, region):
        """Evaluate risk without steering; remove the capture hook on any exit."""
        self._validate(state, region)
        captured = []

        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden[region].float().mean(dim=0))

        handle = self.gate_layer.register_forward_hook(capture)
        try:
            self.model(state)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError('The gate block must execute exactly once per forward')
        hidden = captured[0]
        value = float(((hidden - self.gate_center.to(hidden))
                       @ self.gate_vector.to(hidden)) / self.gate_scale)
        if not math.isfinite(value):
            raise ValueError('Nonfinite gate score')
        return value

    @staticmethod
    def _candidates(region, committed, count):
        runs, run = [], []
        for index in range(len(region) + 1):
            if index < len(region) and region[index]:
                if committed[index]:
                    run.append(index)
            elif run:
                runs.append(run)
                run = []
        if not runs or count <= 0:
            return []
        size = min(count, max(map(len, runs)))
        windows = [run[i:i + size] for run in runs
                   for i in range(0, len(run) - size + 1, size)]
        if len(windows) > 8:
            windows = [windows[round(i * (len(windows) - 1) / 7)] for i in range(8)]
        return windows

    @torch.no_grad()
    def before_step(self, state, region, *, steps_remaining, commit_count):
        self._validate(state, region)
        if steps_remaining <= 0 or commit_count < 0:
            raise ValueError('steps_remaining must be positive and commit_count nonnegative')
        masks = (state == self.mask_id) & region
        if not self.done:
            risky = self.score(state, region) > self.threshold
            if self.mode == 'baseline' or (self.initial_only and self.mode == 'repair'
                                         and self.step == 0 and not risky):
                self.done = True
            if risky:
                self.armed = True
                if self.mode == 'steer':
                    self.done = True
                elif self.mode == 'repair':
                    budget = max(0, steps_remaining * self.max_parallel_commit - int(masks.sum()))
                    candidates = self._candidates(region[0].tolist(), (region & ~masks)[0].tolist(),
                                                  min(self.max_remask_tokens, budget))
                    if candidates:
                        scores = []
                        for positions in candidates:
                            probe = state.clone()
                            probe[0, positions] = self.mask_id
                            scores.append(self.score(probe, region))
                        selected = candidates[min(range(len(scores)), key=scores.__getitem__)]
                        state[0, selected] = self.mask_id
                        masks = (state == self.mask_id) & region
                        remaining = int(masks.sum())
                        commit_count = min(max(1, -(-remaining // steps_remaining)), remaining)
                        self.done = True
        self.step += 1
        return masks, int(masks.sum()), commit_count

    @staticmethod
    def _direction(vector, ratio):
        if vector.ndim == 1:
            return vector
        knots = (0.3, 0.5, 0.7, 0.9)
        high = next((i for i, knot in enumerate(knots) if ratio <= knot), 3)
        low = max(0, high - 1) if ratio < knots[-1] else high
        weight = 0.0 if low == high else (ratio - knots[low]) / (knots[high] - knots[low])
        directions = vector.float()
        mixed = (1 - weight) * directions[low] + weight * directions[high]
        norm = mixed.norm()
        if not torch.isfinite(norm) or norm <= 1e-8:
            raise ValueError('Degenerate interpolated steering direction')
        return mixed / norm

    def _hook(self, vector, positions):
        def steer(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            harmful = -vector.to(hidden)
            unit = harmful / (harmful.norm() + 1e-8)
            values = hidden[positions]
            projection = (values * unit).sum(dim=-1)
            amount = projection.clamp(min=0) + self.strength * (values.norm(dim=-1) + 1e-8)
            updated = hidden.clone()
            updated[positions] = values - amount[:, None] * unit
            return (updated,) + output[1:] if isinstance(output, tuple) else updated
        return steer

    @torch.no_grad()
    def forward(self, state, region, *, positions=None):
        """Apply latched steering; default scope is all hidden-state positions.

        region always describes original answer slots, even when positions
        restricts steering. Hooks are released after success or failure.
        """
        self._validate(state, region)
        if not self.armed:
            return self.model(state)
        positions = torch.ones_like(region) if positions is None else positions
        if positions.shape != state.shape or positions.dtype != torch.bool or positions.device != state.device:
            raise ValueError('Steering positions must be an aligned bool selector')
        ratio = int(((state == self.mask_id) & region).sum()) / int(region.sum())
        handles = []
        try:
            for block, vector in self.sites:
                handles.append(block.register_forward_hook(self._hook(self._direction(vector, ratio), positions)))
            return self.model(state)
        finally:
            for handle in handles:
                handle.remove()