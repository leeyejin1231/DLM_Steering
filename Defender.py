"""Defenses: policy objects the diffusion sampler consults every step.

exp.py selects one by name from DEFENDERS. The whole defended generation runs
through Defender.defend(); internally sampler.generate calls the policy
protocol once per denoising step:

    reset()          per response
    before_step()    may reopen committed answer tokens back to MASK_ID
    forward()        detection + steering inside a single model call
    after_block()    block-boundary hook: audit, remask, or budget extra steps
    result_fields()  per-response record fields

transform_prompt() is a separate prompt-side hook for defenses that only
edit the user message.
"""

import argparse
import math
from abc import ABC, abstractmethod

import torch

from common import (DETECTOR_LAYER, MASK_ID, OUT_DIR, STEER_LAYERS, load_detector,
                    model_blocks)
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

    def after_block(self, x, region, *, block_index, block_positions,
                    prompt_length, temperature, remasking, last_block=False):
        """Called once after each block's denoising loop completes.

        block_positions: answer slots of the just-finished block. prompt_length
        marks the prompt/generation boundary; committed answer slots before it
        (e.g. filled DIJA spans) are also remaskable. last_block marks the
        final block (no later forward exists to piggyback an audit on). May
        audit the block or rewrite committed tokens back to MASK_ID.
        """

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
    """Shared gated-steering machinery for the v2/v3 remask policies.

    A detector hook on blocks[gate-1] reads the current answer and sets a
    continuous gate strength g = clamp((projection - threshold) / width, 0, 1);
    steering hooks on later blocks read g in the same forward and push the
    currently masked answer slots toward refusal.

    --defense ours selects this family and --remask picks the concrete
    policy, built by from_args: V2 (one-shot committed-window repair, the
    llada_steering_remasking_v2 method) or V3 (response-detector boundary
    audit + block recovery). --steer {none,fixed,adaptive} is a separate
    axis: no steering, a step-0 binary gate, or the continuous per-step
    gate. fixed steering reproduces the steer-only measurements of the
    old llada_steering_v2 scripts.
    """

    name = "ours"

    def __init__(self, model, *, gate_layer, gate_vector, threshold, width=1.0,
                 sites, strength=1.0, transform="additive", steer="adaptive",
                 remask_enabled=False, mask_id=MASK_ID):
        """sites: sequence of (layer 1-based, refusal vector [hidden], ref_norm)."""
        if model.training:
            raise ValueError("call model.eval() before constructing the policy")
        if steer not in ("none", "fixed", "adaptive", "triggered"):
            raise ValueError("steer must be none, fixed, adaptive or triggered")
        if steer == "triggered" and not remask_enabled:
            raise ValueError("--steer triggered needs the v3 boundary detector (--remask v3)")
        if transform not in ("additive", "project"):
            raise ValueError("transform must be additive or project")
        if not math.isfinite(threshold) or not math.isfinite(width) or width <= 0:
            raise ValueError("threshold must be finite and width positive")
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        self.steer_enabled = steer != "none"
        # step-0 binary gate only when steering is fixed and no later
        # remask decision needs fresh gate readings
        self.gate_once = steer == "fixed" and not remask_enabled
        self.steer_mode = steer
        blocks = model_blocks(model)
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
        if self.steer_enabled and not self.sites:
            raise ValueError("at least one steering site is required when steering is on")
        self.model = model
        self.gate_layer = gate_layer
        self.gate_block = blocks[gate_layer - 1]
        self.gate_vector = gate_vector.float()
        self.threshold, self.width = float(threshold), float(width)
        self.strength, self.transform = float(strength), transform
        self.mask_id = mask_id
        self.reset()

    @classmethod
    def add_args(cls, parser):
        # Bundle paths and layer defaults follow the selected --model: LLaDA
        # keeps outputs/ + layers 18/25; other models default to their own
        # outputs/<model>/ folder and the layers the fitters chose.
        parser.add_argument("--vector", default=f"{OUT_DIR}/steer_vector.pt")
        parser.add_argument("--detector", default=f"{OUT_DIR}/steer_detector.pt")
        parser.add_argument("--detector-layer", type=int, default=DETECTOR_LAYER,
                            help="Gate layer (hidden-state numbering). Default: 18 for "
                                 "llada, else the detector bundle's best_layer.")
        parser.add_argument("--gate-threshold", type=float, default=None,
                            help="Projection threshold; defaults to gate_threshold.json "
                                 "next to the detector bundle.")
        parser.add_argument("--gate-width", type=float, default=1.0,
                            help="Projection margin above threshold for full steering.")
        parser.add_argument("--layer", default=STEER_LAYERS,
                            help="Comma-separated steering layers (hidden-state numbering; "
                                 "25 = blocks[24]). All must be after --detector-layer. "
                                 "Default: 25 for llada, else the vector bundle's best_layer.")
        parser.add_argument("--alpha", type=float, default=1.0,
                            help="Steering strength in units of the layer's mean activation norm.")
        parser.add_argument("--transform", choices=["additive", "project"], default="additive")
        parser.add_argument("--steer", choices=["none", "fixed", "adaptive", "triggered"],
                            default="adaptive",
                            help="none: no steering; fixed: step-0 binary gate; "
                                 "adaptive: continuous gate every step; triggered: no "
                                 "steering until the v3 response detector fires at a block "
                                 "boundary, then the adaptive gate for the recovery and the "
                                 "rest of the response (requires --remask v3).")
        parser.add_argument("--remask",
                            choices=["none", "v2", "v3"],
                            default="v2",
                            help="none: never remask; v2: one-shot committed-token window "
                                 "repair; v3: response-detector trigger reopens the first "
                                 "block and regenerates it over --recovery-steps steps.")
        parser.add_argument("--response-detector",
                            default=f"{OUT_DIR}/response_detector.pt",
                            help="Logistic-regression response checkpoint; required by "
                                 "--remask v3*. Its layer must match --detector-layer.")
        parser.add_argument("--recovery-steps", type=int, default=32,
                            help="Steps spent regenerating a triggered block (v3).")
        parser.add_argument("--recovery-rounds", type=int, default=1,
                            help="v3: re-audit after each recovery round and remask "
                                 "again while the block still reads as a response, "
                                 "up to this many rounds per trigger.")
        parser.add_argument("--audit-all-boundaries", action="store_true",
                            help="v3: let the response detector trigger at every "
                                 "block boundary, not only the first.")
        parser.add_argument("--recovery-alpha-growth", type=float, default=1.0,
                            help="v3: steering strength multiplier per re-detected "
                                 "recovery round (round i steers at alpha*growth^i).")
        parser.add_argument("--max-remask-tokens", type=int, default=16)
        parser.add_argument("--max-parallel-commit", type=int, default=2)
        parser.add_argument("--remask-trigger", type=float, default=1.0,
                            help="Gate strength needed to attempt the v2 one-shot repair.")

    @classmethod
    def from_args(cls, args, model):
        device = next(model.parameters()).device
        sites = []
        if args.steer != "none":
            bundle = torch.load(args.vector, map_location="cpu")
            layer_spec = (str(bundle["best_layer"]) if args.layer is None
                          else str(args.layer))
            for layer in (int(s) for s in layer_spec.split(",")):
                li = bundle["layers"].index(layer)
                sites.append((layer, bundle["vector"][li].to(device), bundle["mean_act_norm"][li]))
        det_vec, det_layer, threshold = load_detector(
            args.detector, args.detector_layer, device, args.gate_threshold)
        shared = dict(model=model, gate_layer=det_layer, gate_vector=det_vec,
                      threshold=threshold, width=args.gate_width, sites=sites,
                      strength=args.alpha, transform=args.transform,
                      steer=args.steer)
        if args.remask in ("none", "v2"):
            return V2(**shared, remask=args.remask == "v2",
                      max_remask_tokens=args.max_remask_tokens,
                      max_parallel_commit=args.max_parallel_commit,
                      remask_trigger=args.remask_trigger)
        response_detector = torch.load(args.response_detector, map_location="cpu",
                                       weights_only=False)
        return V3(**shared, response_detector=response_detector,
                  recovery_steps=args.recovery_steps,
                  recovery_rounds=args.recovery_rounds,
                  audit_all_boundaries=args.audit_all_boundaries,
                  recovery_alpha_growth=args.recovery_alpha_growth)

    def reset(self):
        self.gate_strength = 0.0
        self.last_projection = None
        self.monitoring = True
        self.step = 0
        self.trace = []
        self._pending = None
        self._schedule_scale = 1.0
        self._steer_boost = 1.0   # V3 recovery rounds may escalate strength
        self.in_recovery = False
        self.triggered = False   # set by V3.after_block when the response detector fires
        self._gate_t = torch.zeros((), dtype=torch.float32,
                                   device=self.gate_vector.device)
        self._pending_audit = None   # V3 defers non-final boundary audits here

    def _steer_armed(self):
        """triggered mode holds steering back until the boundary detector fires."""
        return self.steer_mode != "triggered" or self.triggered

    # ----------------------------------------------------------------- checks
    def _validate(self, x, region, region_count=None):
        nonempty = region.any() if region_count is None else region_count
        if (x.ndim != 2 or x.shape[0] != 1 or x.dtype != torch.long
                or region.shape != x.shape or region.dtype != torch.bool
                or region.device != x.device or not nonempty):
            raise ValueError("expected long x and nonempty bool region shaped [1, seq]")

    def _projection(self, hidden, pool):
        h = hidden[0, pool].to(torch.float32).mean(dim=0)
        value = float(h @ self.gate_vector.to(h.device))
        if not math.isfinite(value):
            raise ValueError("detector returned a non-finite projection")
        return value

    def before_step(self, x, region, *, scope, steps_remaining):
        return None

    # ------------------------------------------------------------------ hooks
    def _gate_hook(self, module, inputs, output):
        pending = self._pending
        if pending is None or pending.get("fired"):
            return output
        hidden = _hidden(output)
        h = hidden[0, pending["pool"]].to(torch.float32).mean(dim=0)
        proj_t = h @ self.gate_vector.to(h.device)
        if self.gate_once:
            g_t = (proj_t >= self.threshold).to(torch.float32) if self.step == 0 else self._gate_t
        else:
            g_t = ((proj_t - self.threshold) / self.width).clamp(0.0, 1.0)
        self._gate_t = g_t
        pending["proj_t"], pending["fired"] = proj_t, True
        return output

    def _steer_hook(self, unit, ref_norm, positions):
        def steer(module, inputs, output):
            pending = self._pending
            live = pending is not None and pending.get("fired")
            if not live and self.gate_strength <= 0.0:
                return output
            if self.strength == 0.0 and self.transform == "additive":
                return output
            hidden = _hidden(output)
            values = hidden[0, positions].to(torch.float32)
            g = (self._gate_t if live else self.gate_strength)
            g = torch.as_tensor(g * self._schedule_scale * self._steer_boost,
                                dtype=torch.float32, device=values.device)
            u = unit.to(values.device)
            if self.transform == "additive":
                updated = values + (g * self.strength * ref_norm) * u
            else:
                harmful = -u
                projection = (values * harmful).sum(dim=-1, keepdim=True).clamp(min=0)
                updated = values - g * (projection + self.strength * ref_norm) * harmful
            out = hidden.clone()
            out[0, positions] = updated.to(hidden.dtype)
            if pending is not None:
                pending["alpha_t"] = g * self.strength
            return _replace(output, out)
        return steer

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, x, region, *, schedule_scale=1.0):
        """One model forward with in-forward detection and steering."""
        masks = (x == self.mask_id) & region
        committed = region & ~masks
        # Sync while the GPU queue is empty: one batched read feeds validate +
        # branch checks + the trace count, and index tensors (not bool masks,
        # which would nonzero-sync mid-forward) are resolved up front.
        n_region, n_masks, n_committed = torch.stack(
            [region.sum(), masks.sum(), committed.sum()]).tolist()
        self._validate(x, region, n_region)
        source = "generated" if n_committed else "masked"
        self._schedule_scale = float(schedule_scale)
        read_gate = self.monitoring and (not self.gate_once or self.step == 0)
        steer = (self.steer_enabled and self.monitoring and n_masks
                 and self._steer_armed() and (read_gate or self.gate_strength > 0.0))
        pool = (committed[0] if n_committed else masks[0]).nonzero().flatten() \
            if read_gate else None
        positions = masks[0].nonzero().flatten() if steer else None
        audit = self._pending_audit     # V3 only: ride this forward's features
        self._pending_audit = None
        audit_pools = None
        if audit is not None:
            audit["chunks"] = self._chunk_positions(
                audit["block_row"].nonzero().flatten())
            audit_pools = [committed[0].nonzero().flatten(), *audit["chunks"]]
        self._pending = {"pool": pool, "fired": False, "proj_t": None, "alpha_t": 0.0}
        handles, feats_out = [], []
        try:
            if read_gate:
                handles.append(self.gate_block.register_forward_hook(self._gate_hook))
            if steer:
                for block, unit, ref_norm in self.sites:
                    handles.append(block.register_forward_hook(
                        self._steer_hook(unit, ref_norm, positions)))
            if audit_pools is not None:
                def capture(module, inputs, output):
                    feats_out.append(torch.stack(
                        [_hidden(output)[0, p].to(torch.float32).mean(dim=0)
                         for p in audit_pools]))
                handles.append(self.gate_block.register_forward_hook(capture))
            output = self.model(x)
        finally:
            for handle in handles:
                handle.remove()
        pend = self._pending
        if read_gate and not pend["fired"]:
            raise RuntimeError("model forward did not execute the gate hook")
        zero = torch.zeros((), dtype=torch.float32, device=x.device)
        scalars = torch.stack([
            pend["proj_t"] if pend["fired"] else zero,
            self._gate_t if pend["fired"] else zero,
            torch.as_tensor(pend["alpha_t"], dtype=torch.float32,
                            device=x.device)])
        if audit is not None:
            if len(feats_out) != 1:
                raise RuntimeError("audit hook must execute exactly once per forward")
            scalars = torch.cat([scalars, self._audit_vector(feats_out[0])])
        vals = scalars.tolist()
        proj_v, gate_v, alpha_v = vals[:3]
        if pend["fired"]:
            if not math.isfinite(proj_v):
                raise ValueError("detector returned a non-finite projection")
            self.gate_strength, self.last_projection = gate_v, proj_v
        if audit is not None and self._apply_audit(
                x, region, self._reading(vals[3:], len(audit["chunks"])),
                audit=audit):
            # The forward just consumed is stale post-recovery; redo it so the
            # sampler commits from post-recovery logits.
            self._pending = None
            return self.forward(x, region, schedule_scale=schedule_scale)
        self.trace.append({
            "step": self.step, "projection": proj_v if pend["fired"] else None,
            "strength": self.gate_strength, "schedule_scale": self._schedule_scale,
            "effective_alpha": alpha_v, "source": source,
            "num_generated_tokens": n_committed,
            "phase": "block_recovery" if self.in_recovery else "base",
            "steer_armed": bool(steer),
        })
        self._pending = None
        self.step += 1
        return output

    def _remasked(self):
        return False

    def result_fields(self):
        if not self.trace:
            return {}
        return {"gate_projection": self.trace[0]["projection"],
                "gate_open": any(t["strength"] > 0 for t in self.trace),
                "gate_max_strength": max(t["strength"] for t in self.trace),
                "remasked": self._remasked(),
                "gate_trace": self.trace}

    def summarize(self, results):
        n_open = sum(r["gate_open"] for r in results)
        n_remask = sum(r["remasked"] for r in results)
        return f"gate opened on {n_open}/{len(results)} prompts, remasked {n_remask}"

    def describe(self):
        return {"defense": self.name, "steer": self.steer_mode,
                "transform": self.transform, "strength": self.strength,
                "threshold": self.threshold, "width": self.width,
                "layers": self.steer_layers}


class V2(Ours):
    """--remask v2: one-shot committed-window repair.

    Once committed answer tokens exist and the gate strength reaches
    --remask-trigger, a single remasking attempt probes candidate windows
    of committed tokens with detector-only forwards and reopens the window
    whose removal lowers the projection the most. This is the
    llada_steering_remasking_v2 method.
    """

    name = "v2"

    def __init__(self, model, *, remask=False, max_remask_tokens=16,
                 max_parallel_commit=2, remask_trigger=1.0, **kw):
        if max_remask_tokens <= 0 or max_parallel_commit <= 0:
            raise ValueError("token budgets must be positive")
        if not 0 < remask_trigger <= 1:
            raise ValueError("remask_trigger must be in (0, 1]")
        if kw.get("steer") == "triggered":
            raise ValueError("--steer triggered needs the v3 boundary detector (--remask v3); "
                             "v2 never raises the trigger, so steering would never start")
        self.remask_enabled = bool(remask)
        self.max_remask_tokens = max_remask_tokens
        self.max_parallel_commit = max_parallel_commit
        self.remask_trigger = float(remask_trigger)
        super().__init__(model, remask_enabled=self.remask_enabled, **kw)

    def reset(self):
        super().reset()
        self.remasked = False
        self.remask_event = None

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
        if (self.remask_enabled and self.monitoring and not self.remasked
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

    def _remasked(self):
        return bool(self.remask_event and self.remask_event["applied"])

    def result_fields(self):
        fields = super().result_fields()
        if fields:
            fields["remask_event"] = self.remask_event
        return fields

    def describe(self):
        d = super().describe()
        d.update(remask="v2" if self.remask_enabled else "none",
                 max_remask_tokens=self.max_remask_tokens,
                 max_parallel_commit=self.max_parallel_commit,
                 remask_trigger=self.remask_trigger)
        return d


class V3(Ours):
    """--remask v3: response-detector boundary audit and block recovery.

    At each block boundary an unsteered audit forward pools gate-layer
    features over the committed tokens and scores them with a
    logistic-regression response detector (--response-detector; its layer
    must match --detector-layer). A trigger on the first block reopens the
    whole block and regenerates it over --recovery-steps dedicated
    forwards.
    """

    name = "v3"

    def __init__(self, model, *, response_detector, recovery_steps=32,
                 recovery_rounds=1, audit_all_boundaries=False,
                 recovery_alpha_growth=1.0, **kw):
        if recovery_steps <= 0:
            raise ValueError("recovery_steps must be positive")
        if recovery_rounds <= 0:
            raise ValueError("recovery_rounds must be positive")
        if not math.isfinite(recovery_alpha_growth) or recovery_alpha_growth <= 0:
            raise ValueError("recovery_alpha_growth must be finite and positive")
        if response_detector is None:
            raise ValueError("--remask v3 requires a response detector "
                             "checkpoint (--response-detector)")
        missing = {"weight", "bias", "threshold", "layer"} - response_detector.keys()
        if missing:
            raise ValueError(f"response detector missing keys: {sorted(missing)}")
        self.recovery_steps = int(recovery_steps)
        self.recovery_rounds = int(recovery_rounds)
        self.audit_all_boundaries = bool(audit_all_boundaries)
        self.recovery_alpha_growth = float(recovery_alpha_growth)
        super().__init__(model, remask_enabled=True, **kw)
        if int(response_detector["layer"]) != self.gate_layer:
            raise ValueError("response detector layer must match the gate layer")
        device = self.gate_vector.device
        self._det_weight = torch.as_tensor(
            response_detector["weight"], dtype=torch.float32, device=device)
        self._det_bias = float(response_detector["bias"])
        self._det_threshold = float(response_detector["threshold"])
        self.response_detector = response_detector

    def reset(self):
        super().reset()
        self.boundary_audits = []
        self.recovery_events = []
        self.audit_forwards = 0

    # --------------------------------------------------- boundary audit/remask
    @staticmethod
    def _chunk_positions(positions, size=32):
        return [positions[i:i + size] for i in range(0, positions.numel(), size)]

    def _audit_vector(self, feats):
        """[committed + chunks] pooled features -> projection/logit/prob vector."""
        logit = feats[0] @ self._det_weight + self._det_bias
        return torch.cat([feats @ self.gate_vector,
                          logit.reshape(1), torch.sigmoid(logit.reshape(1))])

    def _reading(self, values, n_chunks):
        projection = values[0]
        return {
            "projection": projection,
            "block_projections": values[1:1 + n_chunks],
            "response_logit": values[-2],
            "response_probability": values[-1],
            "strength": min(1.0, max(0.0, (projection - self.threshold) / self.width)),
        }

    @torch.no_grad()
    def _audit(self, x, region, chunks):
        """One unsteered forward capturing gate-layer features pooled over the
        committed tokens and each chunk of the finished block."""
        pools = [(region[0] & (x[0] != self.mask_id)).nonzero().flatten(), *chunks]
        feats_out = []

        def capture(module, inputs, output):
            hidden = _hidden(output)
            feats_out.append(torch.stack(
                [hidden[0, p].to(torch.float32).mean(dim=0) for p in pools]))

        handle = self.gate_block.register_forward_hook(capture)
        try:
            self.model(x)
        finally:
            handle.remove()
        self.audit_forwards += 1
        if len(feats_out) != 1:
            raise RuntimeError("audit hook must execute exactly once per forward")
        return self._reading(self._audit_vector(feats_out[0]).tolist(), len(chunks))

    @torch.no_grad()
    def after_block(self, x, region, *, block_index, block_positions,
                    prompt_length, temperature, remasking, last_block=False):
        if self._pending_audit is not None:
            # A deferred audit whose next forward never arrived still runs.
            pending = self._pending_audit
            self._pending_audit = None
            reading = self._audit(x, region, self._chunk_positions(
                pending["block_row"].nonzero().flatten()))
            self._apply_audit(x, region, reading, audit=pending)
        audit = {"block_index": block_index, "block_row": block_positions[0],
                 "prompt_length": prompt_length, "temperature": temperature,
                 "remasking": remasking}
        if last_block:
            # No later forward to piggyback on; audit with a dedicated pass.
            reading = self._audit(x, region, self._chunk_positions(
                block_positions[0].nonzero().flatten()))
            self._apply_audit(x, region, reading, audit=audit)
        else:
            # Deferred: the next block's first defended forward captures the
            # gate-layer features, saving one full forward per clean boundary.
            self._pending_audit = audit

    def _apply_audit(self, x, region, reading, *, audit):
        """Record the audit; run recovery when the block still reads as a
        response. Returns True when recovery ran."""
        block_index = audit["block_index"]
        trigger = ((self.audit_all_boundaries or block_index == 0)
                   and reading["response_probability"] >= self._det_threshold)
        self.boundary_audits.append({
            "boundary": block_index, **reading, "trigger": trigger,
            "trigger_rule": ("response_probability_cutoff_each_boundary"
                             if self.audit_all_boundaries else
                             "response_probability_cutoff_first_boundary")})
        if not trigger:
            return False
        self.triggered = True

        event = {"boundary": block_index,
                 "pre_audit": reading, "pre_recovery_token_ids": x[0].tolist(),
                 "rounds": [], "applied": True,
                 "extra_sampling_steps": self.recovery_steps}
        self.recovery_events.append(event)

        # Remask the finished block plus any committed answer slots inside the
        # prompt (DIJA spans carry the payload under that attack). With
        # recovery_rounds > 1 the block is re-audited after each regeneration
        # and remasked again while it still reads as a response.
        prompt_length, temperature = audit["prompt_length"], audit["temperature"]
        remasking = audit["remasking"]
        span_slots = region[0] & (x[0] != self.mask_id)
        span_slots[prompt_length:] = False
        targets = audit["block_row"] | span_slots
        positions = targets.nonzero().flatten()
        from llada import get_num_transfer_tokens
        from sampler import commit_sample
        counts = get_num_transfer_tokens(
            targets.unsqueeze(0), self.recovery_steps)[0].tolist()
        for round_i in range(self.recovery_rounds):
            # Re-detected rounds steer harder: strength *= growth ** round_i.
            self._steer_boost = self.recovery_alpha_growth ** round_i
            old_tokens = x[0, positions].clone()
            x[0, positions] = self.mask_id
            eligible = positions
            self.in_recovery = True
            try:
                for i in range(self.recovery_steps):
                    final = i == self.recovery_steps - 1
                    if eligible.numel() == 0:
                        break
                    if counts[i] == 0 and not final:
                        continue
                    logits = self.forward(x, region, schedule_scale=1.0).logits
                    eligible = commit_sample(x, logits, eligible, counts[i],
                                             temperature, remasking, final=final)
            finally:
                self.in_recovery = False
            event["rounds"].append({
                "round": round_i,
                "steer_boost": self._steer_boost,
                "selected": positions.tolist(),
                "num_span_positions": int(span_slots.sum()),
                "old_token_ids": old_tokens.tolist(),
                "new_token_ids": x[0, positions].tolist(),
                "round_sampling_forwards": self.recovery_steps,
                "post_trial_token_ids": x[0].tolist()})
            if round_i + 1 >= self.recovery_rounds:
                break
            post = self._audit(x, region, self._chunk_positions(positions))
            self.boundary_audits.append({
                "boundary": block_index, **post, "trigger": False,
                "trigger_rule": "post_recovery_reaudit"})
            if post["response_probability"] < self._det_threshold:
                break
        self._steer_boost = 1.0
        return True

    def _remasked(self):
        return any(e["applied"] for e in self.recovery_events)

    def result_fields(self):
        fields = super().result_fields()
        if fields and self.boundary_audits:
            fields.update(boundary_audits=self.boundary_audits,
                          recovery_events=self.recovery_events,
                          audit_forwards=self.audit_forwards)
        return fields

    def describe(self):
        d = super().describe()
        d.update(remask="v3", recovery_steps=self.recovery_steps,
                 recovery_rounds=self.recovery_rounds,
                 audit_all_boundaries=self.audit_all_boundaries,
                 recovery_alpha_growth=self.recovery_alpha_growth)
        return d


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
        parser.add_argument("--vector", default=f"{OUT_DIR}/steer_vector.pt")
        parser.add_argument("--detector", default=f"{OUT_DIR}/steer_detector.pt")
        parser.add_argument("--detector-layer", type=int, default=DETECTOR_LAYER)
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

    def defend(self, model, prompt_ids, **gen_config):
        # Attacks may override steps per prompt (DIJA: one mask per step).
        self.total_steps = gen_config.get("steps", self.total_steps)
        return super().defend(model, prompt_ids, **gen_config)

    def reset(self):
        self._policy = Proposed.from_llada(self.model, self.gate, self.csd,
                                           layers=self.layers, mask_id=MASK_ID,
                                           **self.options)
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
