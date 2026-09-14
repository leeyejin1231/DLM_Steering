"""Defenses: policy objects the diffusion sampler consults every step.

exp.py selects one by name from DEFENDERS. The whole defended generation runs
through Defender.defend(); internally sampler.generate calls the policy
protocol once per denoising step:

    reset()          per response
    before_step()    may reopen committed answer tokens back to MASK_ID
    forward()        detection + steering inside a single model call
    result_fields()  per-response record fields

transform_prompt() is a separate prompt-side hook for defenses that only
edit the user message.
"""

import math
from abc import ABC, abstractmethod

import torch

from common import MASK_ID, load_detector
from proposed import Proposed


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace(output, hidden):
    return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


class Defender(ABC):
    """Per-run defense policy; the sampler owns token sampling."""

    name: str

    @classmethod
    def add_args(cls, parser):
        """Register defense-specific CLI arguments (optional)."""

    @classmethod
    @abstractmethod
    def from_args(cls, args, model):
        """Build the policy: load bundles, construct, return the instance."""

    def transform_prompt(self, user_message):
        """Prompt-side hook; identity for activation-level defenses."""
        return user_message

    @abstractmethod
    def reset(self):
        """Reset per-response state."""

    @abstractmethod
    def before_step(self, x, region, *, scope, steps_remaining):
        """May rewrite committed answer tokens in x back to MASK_ID, in place.

        region: all original answer slots (bool [1, seq]). scope: slots that can
        still be filled in the current block (positions < block_end).
        steps_remaining: denoising steps left in this block.
        Returns the commit count for this step if masks were reopened (the
        sampler rebalances its schedule), else None.
        """

    @abstractmethod
    def forward(self, x, region, *, schedule_scale):
        """One model forward, optionally with in-forward detection + steering.
        Must return the model output (with .logits)."""

    @abstractmethod
    def result_fields(self):
        """Fields merged into this response's result record."""

    def summarize(self, results):
        """One-line run summary, or None."""
        return None

    def describe(self):
        """Config dict stored in the result payload."""
        return {"defense": self.name}

    def defend(self, model, prompt_ids, **gen_config):
        """Run one defended generation through the unified sampler."""
        from sampler import generate
        return generate(model, prompt_ids, self, **gen_config)


class NullDefender(Defender):
    """Undefended reference: plain model forwards."""

    name = "none"

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_args(cls, args, model):
        return cls(model)

    def reset(self):
        pass

    def before_step(self, x, region, *, scope, steps_remaining):
        return None

    def forward(self, x, region, *, schedule_scale):
        return self.model(x)

    def result_fields(self):
        return {}


class Ours(Defender):
    """Per-response state for gated steering and one-shot remasking.

    A detector hook on blocks[gate-1] reads the current answer and sets a
    continuous gate strength g = clamp((projection - threshold) / width, 0, 1);
    steering hooks on later blocks read g in the same forward and push the
    currently masked answer slots toward refusal. Once committed answer tokens
    exist and g >= remask_trigger, a single remasking attempt probes candidate
    windows of committed tokens with detector-only forwards and reopens the
    window whose removal lowers the projection the most.

    Call reset() before each response. before_step may rewrite committed
    answer tokens back to MASK_ID in place.
    """

    name = "ours"

    def __init__(self, model, *, gate_layer, gate_vector, threshold, width=1.0,
                 sites, strength=1.0, transform="additive", mode="repair",
                 max_remask_tokens=16, max_parallel_commit=2, remask_trigger=1.0,
                 initial_only=False, mask_id=MASK_ID):
        """sites: sequence of (layer 1-based, refusal vector [hidden], ref_norm)."""
        if model.training:
            raise ValueError("call model.eval() before constructing the policy")
        if mode not in ("baseline", "steer", "repair"):
            raise ValueError("mode must be baseline, steer or repair")
        if transform not in ("additive", "project"):
            raise ValueError("transform must be additive or project")
        if not math.isfinite(threshold) or not math.isfinite(width) or width <= 0:
            raise ValueError("threshold must be finite and width positive")
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        if max_remask_tokens <= 0 or max_parallel_commit <= 0:
            raise ValueError("token budgets must be positive")
        if not 0 < remask_trigger <= 1:
            raise ValueError("remask_trigger must be in (0, 1]")
        blocks = model.model.transformer.blocks
        if not 1 <= gate_layer <= len(blocks):
            raise ValueError(f"gate layer must be in 1..{len(blocks)}")
        if gate_vector.ndim != 1 or not torch.isfinite(gate_vector).all() or gate_vector.norm() == 0:
            raise ValueError("gate vector must be a finite nonzero 1D tensor")
        self.sites = []
        self.steer_layers = []
        for layer, vector, ref_norm in sites:
            if not gate_layer < layer <= len(blocks):
                raise ValueError(f"steering layer {layer} must be after the gate layer "
                                 f"{gate_layer} and within 1..{len(blocks)}")
            if vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm() == 0:
                raise ValueError("steering vectors must be finite nonzero 1D tensors")
            if not math.isfinite(ref_norm) or ref_norm <= 0:
                raise ValueError("reference norm must be finite and positive")
            self.sites.append((blocks[layer - 1], vector.float() / vector.norm(), float(ref_norm)))
            self.steer_layers.append(layer)
        if not self.sites:
            raise ValueError("at least one steering site is required")
        self.model = model
        self.gate_block = blocks[gate_layer - 1]
        self.gate_vector = gate_vector.float()
        self.threshold, self.width = float(threshold), float(width)
        self.strength, self.transform, self.mode = float(strength), transform, mode
        self.max_remask_tokens, self.max_parallel_commit = max_remask_tokens, max_parallel_commit
        self.remask_trigger, self.initial_only = float(remask_trigger), initial_only
        self.mask_id = mask_id
        self.reset()

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--vector", default="outputs/steer_vector.pt")
        parser.add_argument("--detector", default="outputs/steer_detector.pt")
        parser.add_argument("--detector-layer", type=int, default=18)
        parser.add_argument("--gate-threshold", type=float, default=None,
                            help="Projection threshold; defaults to gate_threshold.json "
                                 "next to the detector bundle.")
        parser.add_argument("--gate-width", type=float, default=1.0,
                            help="Projection margin above threshold for full steering.")
        parser.add_argument("--layer", default="25",
                            help="Comma-separated steering layers (hidden-state numbering; "
                                 "25 = blocks[24]). All must be after --detector-layer.")
        parser.add_argument("--alpha", type=float, default=1.0,
                            help="Steering strength in units of the layer's mean activation norm.")
        parser.add_argument("--transform", choices=["additive", "project"], default="additive")
        parser.add_argument("--mode", choices=["baseline", "steer", "repair"], default="repair",
                            help="baseline: one-time binary gate; steer: adaptive gate; "
                                 "repair: adaptive gate + one-shot remask.")
        parser.add_argument("--max-remask-tokens", type=int, default=16)
        parser.add_argument("--max-parallel-commit", type=int, default=2)
        parser.add_argument("--remask-trigger", type=float, default=1.0,
                            help="Gate strength needed to attempt the one-shot repair.")
        parser.add_argument("--initial-only", action="store_true",
                            help="Stop monitoring for the whole response if the step-0 gate is closed.")

    @classmethod
    def from_args(cls, args, model):
        device = next(model.parameters()).device
        bundle = torch.load(args.vector, map_location="cpu")
        sites = []
        for layer in (int(s) for s in args.layer.split(",")):
            li = bundle["layers"].index(layer)
            sites.append((layer, bundle["vector"][li].to(device), bundle["mean_act_norm"][li]))
        det_vec, det_layer, threshold = load_detector(
            args.detector, args.detector_layer, device, args.gate_threshold)
        return cls(model, gate_layer=det_layer, gate_vector=det_vec,
                   threshold=threshold, width=args.gate_width, sites=sites,
                   strength=args.alpha, transform=args.transform, mode=args.mode,
                   max_remask_tokens=args.max_remask_tokens,
                   max_parallel_commit=args.max_parallel_commit,
                   remask_trigger=args.remask_trigger, initial_only=args.initial_only)

    def reset(self):
        self.gate_strength = 0.0
        self.last_projection = None
        self.monitoring = True
        self.remasked = False
        self.step = 0
        self.trace = []
        self.remask_event = None
        self._pending = None
        self._schedule_scale = 1.0

    # ----------------------------------------------------------------- checks
    def _validate(self, x, region):
        if (x.ndim != 2 or x.shape[0] != 1 or x.dtype != torch.long
                or region.shape != x.shape or region.dtype != torch.bool
                or region.device != x.device or not region.any()):
            raise ValueError("expected long x and nonempty bool region shaped [1, seq]")

    def _projection(self, hidden, pool):
        h = hidden[0, pool].to(torch.float32).mean(dim=0)
        value = float(h @ self.gate_vector.to(h.device))
        if not math.isfinite(value):
            raise ValueError("detector returned a non-finite projection")
        return value

    # ---------------------------------------------------------------- probes
    @torch.no_grad()
    def score(self, x, pool):
        """Detector-only forward (no steering). pool: bool [seq] positions to average."""
        captured = []

        def capture(module, inputs, output):
            captured.append(self._projection(_hidden(output), pool))

        handle = self.gate_block.register_forward_hook(capture)
        try:
            self.model(x)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("gate block must execute exactly once per forward")
        return captured[0]

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

    def before_step(self, x, region, *, scope, steps_remaining):
        """One-shot repair using the previous forward's gate reading."""
        self._validate(x, region)
        if scope.shape != x.shape or scope.dtype != torch.bool or steps_remaining <= 0:
            raise ValueError("scope must be an aligned bool mask and steps_remaining positive")
        masks = (x == self.mask_id) & region
        commit_count = None
        if (self.mode == "repair" and self.monitoring and not self.remasked
                and self.step > 0 and self.gate_strength >= self.remask_trigger):
            committed = region & ~masks & scope
            budget = max(0, steps_remaining * self.max_parallel_commit - int((masks & scope).sum()))
            candidates = self._candidates(region[0].tolist(), committed[0].tolist(),
                                          min(self.max_remask_tokens, budget))
            if candidates:
                self.remasked = True
                pool = committed[0].clone()  # fixed across probes so scores are comparable
                base = self.score(x, pool)
                scores = []
                for positions in candidates:
                    probe = x.clone()
                    probe[0, positions] = self.mask_id
                    scores.append(self.score(probe, pool))
                best = min(range(len(scores)), key=scores.__getitem__)
                applied = scores[best] < base
                event = {"step": self.step, "gate_strength": self.gate_strength,
                         "base_projection": base, "candidate_projections": scores,
                         "candidates": candidates, "applied": applied,
                         "selected": candidates[best] if applied else None}
                if applied:
                    x[0, candidates[best]] = self.mask_id
                    masks = (x == self.mask_id) & region
                    remaining = int((masks & scope).sum())
                    commit_count = min(max(1, -(-remaining // steps_remaining)), remaining)
                    event["commit_count"] = commit_count
                self.remask_event = event
        return commit_count

    # ------------------------------------------------------------------ hooks
    def _gate_hook(self, module, inputs, output):
        pending = self._pending
        if pending is None or pending.get("fired"):
            return output
        projection = self._projection(_hidden(output), pending["pool"])
        if self.mode == "baseline":
            g = float(projection >= self.threshold) if self.step == 0 else self.gate_strength
        else:
            g = min(1.0, max(0.0, (projection - self.threshold) / self.width))
        self.gate_strength, self.last_projection = g, projection
        pending["projection"], pending["fired"] = projection, True
        return output

    def _steer_hook(self, unit, ref_norm, positions):
        def steer(module, inputs, output):
            g = self.gate_strength * self._schedule_scale
            if g <= 0.0 or (self.strength == 0.0 and self.transform == "additive"):
                return output
            hidden = _hidden(output)
            values = hidden[0, positions].to(torch.float32)
            u = unit.to(values.device)
            if self.transform == "additive":
                updated = values + (g * self.strength * ref_norm) * u
            else:
                harmful = -u
                projection = (values * harmful).sum(dim=-1, keepdim=True).clamp(min=0)
                updated = values - g * (projection + self.strength * ref_norm) * harmful
            out = hidden.clone()
            out[0, positions] = updated.to(hidden.dtype)
            self._pending["effective_alpha"] = g * self.strength
            return _replace(output, out)
        return steer

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, x, region, *, schedule_scale=1.0):
        """One model forward with in-forward detection and steering."""
        self._validate(x, region)
        masks = (x == self.mask_id) & region
        committed = region & ~masks
        source = "generated" if committed.any() else "masked"
        pool = committed[0] if committed.any() else masks[0]
        self._schedule_scale = float(schedule_scale)
        read_gate = self.monitoring and (self.mode != "baseline" or self.step == 0)
        steer = self.monitoring and masks.any() and (read_gate or self.gate_strength > 0.0)
        self._pending = {"pool": pool, "fired": False, "projection": None, "effective_alpha": 0.0}
        handles = []
        try:
            if read_gate:
                handles.append(self.gate_block.register_forward_hook(self._gate_hook))
            if steer:
                for block, unit, ref_norm in self.sites:
                    handles.append(block.register_forward_hook(self._steer_hook(unit, ref_norm, masks[0])))
            output = self.model(x)
        finally:
            for handle in handles:
                handle.remove()
        if read_gate and not self._pending["fired"]:
            raise RuntimeError("model forward did not execute the gate hook")
        self.trace.append({
            "step": self.step, "projection": self._pending["projection"],
            "strength": self.gate_strength, "schedule_scale": self._schedule_scale,
            "effective_alpha": self._pending["effective_alpha"], "source": source,
            "num_generated_tokens": int(committed.sum()),
        })
        if self.step == 0 and self.initial_only and self.gate_strength == 0.0:
            self.monitoring = False
        self._pending = None
        self.step += 1
        return output

    def result_fields(self):
        if not self.trace:
            return {}
        return {"gate_projection": self.trace[0]["projection"],
                "gate_open": any(t["strength"] > 0 for t in self.trace),
                "gate_max_strength": max(t["strength"] for t in self.trace),
                "remasked": bool(self.remask_event and self.remask_event["applied"]),
                "remask_event": self.remask_event,
                "gate_trace": self.trace}

    def summarize(self, results):
        n_open = sum(r["gate_open"] for r in results)
        n_remask = sum(r["remasked"] for r in results)
        return f"gate opened on {n_open}/{len(results)} prompts, remasked {n_remask}"

    def describe(self):
        return {"defense": self.name, "mode": self.mode, "transform": self.transform,
                "strength": self.strength, "threshold": self.threshold, "width": self.width,
                "layers": self.steer_layers, "max_remask_tokens": self.max_remask_tokens,
                "max_parallel_commit": self.max_parallel_commit,
                "remask_trigger": self.remask_trigger, "initial_only": self.initial_only}


class ProposedDefense(Defender):
    """proposed.py's single-file defense, adapted to the Defender protocol.

    proposed.py wants one instance per response, so reset() builds a fresh
    Proposed; its before_step is block-agnostic and budgets against the whole
    generation, so it is driven with total-step accounting here.
    """

    name = "proposed"

    def __init__(self, model, gate, csd, *, layers, total_steps, **options):
        self.model, self.gate, self.csd = model, gate, csd
        self.layers, self.options, self.total_steps = layers, options, total_steps
        self._policy = None
        self.step, self.trace, self.remask_event = 0, [], None

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--vector", default="outputs/steer_vector.pt")
        parser.add_argument("--detector", default="outputs/steer_detector.pt")
        parser.add_argument("--detector-layer", type=int, default=18)
        parser.add_argument("--gate-threshold", type=float, default=None)
        parser.add_argument("--layers", default="12,16,20,24")
        parser.add_argument("--strength", type=float, default=0.4)
        parser.add_argument("--mode", choices=["baseline", "steer", "repair"], default="repair")
        parser.add_argument("--max-remask-tokens", type=int, default=16)
        parser.add_argument("--max-parallel-commit", type=int, default=2)
        parser.add_argument("--no-initial-only", action="store_true")

    @classmethod
    def from_args(cls, args, model):
        device = next(model.parameters()).device
        csd = torch.load(args.vector, map_location="cpu")
        det_vec, det_layer, threshold = load_detector(
            args.detector, args.detector_layer, device, args.gate_threshold)
        gate = {"layer": det_layer, "vector": det_vec,
                "center": torch.zeros_like(det_vec), "scale": 1.0, "threshold": threshold}
        csd = {"layers": csd["layers"], "vector": csd["vector"].to(device)}
        layers = tuple(int(s) for s in args.layers.split(","))
        return cls(model, gate, csd, layers=layers, total_steps=args.steps,
                   strength=args.strength, mode=args.mode,
                   max_remask_tokens=args.max_remask_tokens,
                   max_parallel_commit=args.max_parallel_commit,
                   initial_only=not args.no_initial_only)

    def reset(self):
        self._policy = Proposed.from_llada(self.model, self.gate, self.csd,
                                           layers=self.layers, **self.options)
        self.step, self.trace, self.remask_event = 0, [], None

    def before_step(self, x, region, *, scope, steps_remaining):
        before = int(((x == MASK_ID) & region).sum())
        masks, _, _ = self._policy.before_step(
            x, region, steps_remaining=self.total_steps - self.step, commit_count=1)
        reopened = int(masks.sum()) - before
        commit_count = None
        if reopened > 0:
            self.remask_event = {"step": self.step, "reopened": reopened}
            commit_count = 1  # any value: the sampler rebalances the block schedule
        self.trace.append({"step": self.step, "armed": self._policy.armed,
                           "done": self._policy.done})
        return commit_count

    def forward(self, x, region, *, schedule_scale):
        self.step += 1
        return self._policy.forward(x, region)

    def result_fields(self):
        return {"gate_open": self._policy.armed,
                "armed_at_step": next((t["step"] for t in self.trace if t["armed"]), None),
                "monitoring_steps": sum(1 for t in self.trace if not t["done"]),
                "remasked": self.remask_event is not None,
                "remask_event": self.remask_event}

    def summarize(self, results):
        n_open = sum(r["gate_open"] for r in results)
        n_remask = sum(r["remasked"] for r in results)
        return f"gate opened on {n_open}/{len(results)} prompts, remasked {n_remask}"


class SelfReminder(NullDefender):
    """Prompt-side baseline: prepends a responsibility reminder to the user turn.
    Model behaviour is untouched (plain forwards)."""

    name = "selfreminder"
    DEFAULT_REMINDER = ("You should be a responsible AI assistant and must not "
                        "generate harmful, misleading, or illegal content.")

    def __init__(self, model, reminder=None):
        super().__init__(model)
        self.reminder = reminder or self.DEFAULT_REMINDER

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--reminder-text", default=None,
                            help="Reminder sentence prepended to the user turn.")

    @classmethod
    def from_args(cls, args, model):
        return cls(model, args.reminder_text)

    def transform_prompt(self, user_message):
        return f"{self.reminder}\n\n{user_message}"

    def describe(self):
        return {"defense": self.name, "reminder": self.reminder}


class DiffuGuard(Defender):
    """Diffusion-time defense baseline; not implemented yet."""

    name = "diffuguard"

    @classmethod
    def from_args(cls, args, model):
        raise NotImplementedError("DiffuGuard not implemented yet")

    def reset(self):
        pass

    def before_step(self, x, region, *, scope, steps_remaining):
        return None

    def forward(self, x, region, *, schedule_scale):
        raise NotImplementedError("DiffuGuard not implemented yet")

    def result_fields(self):
        return {}


DEFENDERS = {d.name: d for d in (NullDefender, Ours, ProposedDefense, SelfReminder, DiffuGuard)}
