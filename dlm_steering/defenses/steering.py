"""Detection and gated activation steering during each model forward."""
import math
import torch
from dlm_steering.runtime.constants import (
    MASK_ID, MODEL, MODEL_LOCK, OUT_DIR, DETECTOR_LAYER, STEER_LAYERS, block_index,
)
from dlm_steering.runtime.models import load_detector, model_blocks
from sampler import take_true
from .base import Defender, _ForwardPlan, _GatePass, _StepScalars, _hidden, _replace


class Ours(Defender):
    """Gated steering, optionally extended by V3 boundary recovery.

    A detector hook on blocks[gate-1] reads the current answer and sets a
    continuous gate strength g = clamp((projection - threshold) / width, 0, 1);
    steering hooks on later blocks read g in the same forward and push the
    currently masked answer slots toward refusal.

    --defense ours selects this family and --remask picks the concrete
    policy, built by from_args: Ours (steering without remasking) or V3
    (response-detector boundary audit + block recovery). --steer {none,fixed,adaptive} is a separate
    axis: no steering, a step-0 binary gate, or the continuous per-step
    gate. fixed steering is the steer-only configuration (gate read once,
    never remask) that the earlier standalone steering scripts measured.
    """

    name = "ours"

    def __init__(self, model, *, gate_layer, gate_vector, threshold, width=1.0,
                 sites, strength=1.0, transform="additive", steer="adaptive",
                 remask_enabled=False, mask_id=MASK_ID, steer_shift=False):
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
            self.sites.append((blocks[block_index(layer)],
                               vector.float() / vector.norm(), float(ref_norm)))
            self.steer_layers.append(layer)
        if self.steer_enabled and not self.sites:
            raise ValueError("at least one steering site is required when steering is on")
        self.model = model
        self.gate_layer = gate_layer
        self.gate_block = blocks[block_index(gate_layer)]
        self.ln_f = None if MODEL["shift_logits"] else model.model.transformer.ln_f
        self.gate_vector = gate_vector.float()
        self.threshold, self.width = float(threshold), float(width)
        self.strength, self.transform = float(strength), transform
        self.mask_id = mask_id
        self.steer_shift = steer_shift
        self.reset()

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--vector", default=f"{OUT_DIR}/steer_vector.pt")
        parser.add_argument("--detector", default=f"{OUT_DIR}/steer_detector.pt")
        parser.add_argument("--detector-layer", type=int, default=DETECTOR_LAYER)
        parser.add_argument("--gate-threshold", type=float, default=None,
                            help="Projection threshold; defaults to gate_threshold.json "
                                 "next to the detector bundle.")
        parser.add_argument("--gate-width", type=float, default=1.0,
                            help="Projection margin above threshold for full steering.")
        parser.add_argument("--layer", default=STEER_LAYERS,
                            help="Comma-separated steering layers (hidden-state numbering; "
                                 "25 = blocks[24]). All must be after --detector-layer.")
        parser.add_argument("--alpha", type=float, default=1.0,
                            help="Steering strength in units of the layer's mean activation norm.")
        parser.add_argument("--transform", choices=["additive", "project"], default="additive")
        parser.add_argument("--steer-shift", action="store_true",
                            help="Also steer the position before masked slots (Dream).")
        parser.add_argument("--steer", choices=["none", "fixed", "adaptive", "triggered"],
                            default="adaptive",
                            help="none: no steering; fixed: step-0 binary gate; "
                                 "adaptive: continuous gate every step; triggered: no "
                                 "steering until the v3 response detector fires at a block "
                                 "boundary, then the adaptive gate for the recovery and the "
                                 "rest of the response (requires --remask v3).")
        parser.add_argument("--remask",
                            choices=["none", "v3"],
                            default="v3",
                            help="none: never remask; v3: response-detector trigger reopens the first "
                                 "block and regenerates it over --recovery-steps steps.")
        parser.add_argument("--response-detector", default=f"{OUT_DIR}/response_detector.pt",
                            help="Logistic-regression response checkpoint; required by "
                                 "--remask v3*. Its layer must match --detector-layer.")
        parser.add_argument("--response-threshold", type=float, default=None,
                            help="Override the response detector's own trigger "
                                 "cutoff. The checkpoint stores the threshold it "
                                 "was fit with; this retunes the operating point "
                                 "without refitting.")
        parser.add_argument("--remask-prompt", action="store_true",
                            help="v3: recover all prompt text, preserving special tokens.")
        parser.add_argument("--recovery-steps", type=lambda s: "auto" if s == "auto" else int(s), default=32,
                            help="Steps spent regenerating a triggered block (v3).")
        parser.add_argument("--recovery-rounds", type=int, default=1,
                            help="v3: re-audit after each recovery round and remask "
                                 "again while the block still reads as a response, "
                                 "up to this many rounds per trigger.")
        parser.add_argument("--audit-all-boundaries", action="store_true",
                            help="v3: let the response detector trigger at every "
                                 "block boundary, not only --audit-boundary.")
        parser.add_argument("--infill-checkpoint", type=int, default=0,
                            help="v3, gen_length 0 (DIJA): audit after every N "
                                 "committed mask slots, so --audit-boundary k "
                                 "means after N*(k+1) slots. The fill order is "
                                 "unchanged. A row with fewer slots never reaches "
                                 "the later checkpoints and gets no recovery there. "
                                 "0 (default) audits once, when infilling ends.")
        parser.add_argument("--audit-boundary", type=int, default=0,
                            help="v3: the block boundary (0-based: 0 = after the "
                                 "first block) at which the response detector may "
                                 "trigger. Every boundary is still audited and "
                                 "recorded. Needs that many blocks to exist: with "
                                 "gen_length 0 (DIJA infilling) there is only "
                                 "boundary 0. Ignored under --audit-all-boundaries.")
        parser.add_argument("--recovery-alpha-growth", type=float, default=1.0,
                            help="v3: steering strength multiplier per re-detected "
                                 "recovery round (round i steers at alpha*growth^i).")

    @classmethod
    def from_args(cls, args, model):
        if args.remask_prompt and args.remask != "v3":
            raise ValueError("--remask-prompt requires --remask v3")
        device = next(model.parameters()).device
        sites = []
        if args.steer != "none":
            bundle = torch.load(args.vector, map_location="cpu")
            layer_spec = str(bundle["best_layer"]) if args.layer is None else args.layer
            for layer in (int(s) for s in layer_spec.split(",")):
                li = bundle["layers"].index(layer)
                sites.append((layer, bundle["vector"][li].to(device), bundle["mean_act_norm"][li]))
        det_vec, det_layer, threshold = load_detector(
            args.detector, args.detector_layer, device, args.gate_threshold)
        shared = dict(model=model, gate_layer=det_layer, gate_vector=det_vec,
                      threshold=threshold, width=args.gate_width, sites=sites,
                      strength=args.alpha, transform=args.transform,
                      steer=args.steer, steer_shift=args.steer_shift)
        if args.remask == "none":
            return cls(**shared)
        response_detector = torch.load(args.response_detector, map_location="cpu",
                                       weights_only=False)
        if args.response_threshold is not None:
            if not 0.0 < args.response_threshold < 1.0:
                raise ValueError("--response-threshold must lie in (0, 1)")
            response_detector = {**response_detector,
                                 "threshold": float(args.response_threshold)}
        from .recovery import V3
        return V3(**shared, response_detector=response_detector,
                  remask_prompt=args.remask_prompt,
                  recovery_steps=args.recovery_steps,
                  recovery_rounds=args.recovery_rounds,
                  audit_all_boundaries=args.audit_all_boundaries,
                  audit_boundary=args.audit_boundary,
                  infill_checkpoint=args.infill_checkpoint,
                  recovery_alpha_growth=args.recovery_alpha_growth)

    def reset(self):
        self.gate_strength = 0.0
        self.last_projection = None
        self.monitoring = True
        self.step = 0
        self.trace = []
        self._trace_raw = []     # per-step rows whose numbers are still on the GPU
        self._n_region = None    # region size, read once per response
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

    # ------------------------------------------------------------------ hooks
    def _gate_hook(self, module, inputs, output):
        """Project the current answer onto the detector direction, set the gate.

        Runs on blocks[gate_layer - 1], so every steering hook later in the
        same forward reads a gate strength computed from this same step.
        """
        pending = self._pending
        if pending is None or pending.fired:
            return output
        hidden = _hidden(output)
        h = hidden[0, pending.pool].to(torch.float32).mean(dim=0)
        proj_t = h @ self.gate_vector.to(h.device)
        if self.gate_once:
            g_t = (proj_t >= self.threshold).to(torch.float32) if self.step == 0 else self._gate_t
        else:
            g_t = ((proj_t - self.threshold) / self.width).clamp(0.0, 1.0)
        self._gate_t = g_t
        pending.projection, pending.fired = proj_t, True
        return output

    def _audit_capture_hook(self, pools, out):
        """Gate-layer hook: mean-pool hidden states over each index tensor."""
        def capture(module, inputs, output):
            out.append(torch.stack(
                [_hidden(output)[0, p].to(torch.float32).mean(dim=0)
                 for p in pools]))
        return capture

    def _steer_hook(self, unit, ref_norm, mask):
        """mask: bool [seq], the slots to push. The update is computed for every
        position and kept only where mask is set -- an indexed write would cost
        two GPU syncs per call under --reproduct's deterministic mode. The
        additive update is pointwise, so masked slots get exactly the values the
        indexed version produced."""
        def steer(module, inputs, output):
            pending = self._pending
            live = pending is not None and pending.fired
            if not live and self.gate_strength <= 0.0:
                return output
            if self.strength == 0.0 and self.transform == "additive":
                return output
            hidden = _hidden(output)
            values = hidden[0].to(torch.float32)
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
            out = torch.where(mask[:, None], updated.to(hidden.dtype), hidden[0])[None]
            if pending is not None:
                pending.alpha = g * self.strength
            return _replace(output, out)
        return steer

    # ------------------------------------------------------- forward machinery
    # Layout of the single batched scalar read: projection, gate strength,
    # effective alpha, then the audit vector when an audit rides this forward.
    _N_STEP_SCALARS = 3

    def _plan_forward(self, x, region, n_masks=None):
        """Decide what this forward reads, steers and audits.

        All index tensors are resolved here -- a nonzero() inside a hook would
        sync the GPU in the middle of the model call. Given the open-mask count
        (the sampler passes it), nothing here waits on the GPU: the region size
        is read once per response and index sets are cut to their known size
        with take_true instead of nonzero().
        """
        masks = (x == self.mask_id) & region
        committed = region & ~masks
        if n_masks is None:
            n_region, n_masks, n_committed = torch.stack(
                [region.sum(), masks.sum(), committed.sum()]).tolist()
            self._validate(x, region, n_region)
        else:
            if self._n_region is None:
                self._n_region = int(region.sum())
                self._validate(x, region, self._n_region)
            n_committed = self._n_region - n_masks

        read_gate = self.monitoring and (not self.gate_once or self.step == 0)
        steer = bool(self.steer_enabled and self.monitoring and n_masks
                     and self._steer_armed()
                     and (read_gate or self.gate_strength > 0.0))
        # Before anything is committed the gate has to read the masked slots
        # themselves; after that it reads what the model actually wrote.
        gate_pool = None
        if read_gate:
            gate_pool = (take_true(committed[0], n_committed) if n_committed
                         else take_true(masks[0], n_masks))

        audit, audit_pools = self._pending_audit, None
        self._pending_audit = None
        if audit is not None:
            audit.chunks = self._chunk_positions(
                audit.block_row.nonzero().flatten())
            audit_pools = [take_true(committed[0], n_committed), *audit.chunks]

        steer_mask = masks[0] if steer else None
        if steer and self.steer_shift:
            steer_mask = steer_mask.clone()
            steer_mask[:-1] |= masks[0, 1:]

        return _ForwardPlan(
            source="generated" if n_committed else "masked",
            n_committed=n_committed, read_gate=read_gate, steer=steer,
            gate_pool=gate_pool,
            steer_mask=steer_mask,
            audit=audit, audit_pools=audit_pools)

    def _run_hooked_forward(self, x, plan, logit_positions):
        """One model call with this step's hooks; they are always removed.

        Returns (model output, captured audit features). Forward hooks are
        module-global, so MODEL_LOCK spans the whole register/call/remove
        window -- see common.MODEL_LOCK.
        """
        handles, feats = [], []
        with MODEL_LOCK:
            try:
                if plan.read_gate:
                    handles.append(self.gate_block.register_forward_hook(
                        self._gate_hook))
                if plan.steer:
                    for block, unit, ref_norm in self.sites:
                        handles.append(block.register_forward_hook(
                            self._steer_hook(unit, ref_norm, plan.steer_mask)))
                if plan.audit_pools is not None:
                    handles.append(self.gate_block.register_forward_hook(
                        self._audit_capture_hook(plan.audit_pools, feats)))
                if logit_positions is not None and self.ln_f is not None:
                    # Slice ln_f so the vocab projection runs only on the rows
                    # the sampler is about to read.
                    handles.append(self.ln_f.register_forward_hook(
                        lambda m, i, o: o[:, logit_positions]))
                output = self.model(x)
            finally:
                for handle in handles:
                    handle.remove()
        if plan.read_gate and not self._pending.fired:
            raise RuntimeError("model forward did not execute the gate hook")
        if plan.audit_pools is not None and len(feats) != 1:
            raise RuntimeError("audit hook must execute exactly once per forward")
        return output, feats

    def _step_tensor(self, x):
        """This step's projection, gate strength and effective alpha, on the GPU."""
        pending = self._pending
        zero = torch.zeros((), dtype=torch.float32, device=x.device)
        return torch.stack([
            pending.projection if pending.fired else zero,
            self._gate_t if pending.fired else zero,
            torch.as_tensor(pending.alpha, dtype=torch.float32,
                            device=x.device)])

    def _read_scalars(self, plan, feats, step_t):
        """Sync the step scalars (and the audit vector, if any) in ONE .tolist()."""
        scalars = step_t
        if plan.audit is not None:
            scalars = torch.cat([scalars, self._audit_vector(feats[0])])
        values = scalars.tolist()
        projection, gate_strength, alpha = values[:self._N_STEP_SCALARS]
        return _StepScalars(projection, gate_strength, alpha,
                            values[self._N_STEP_SCALARS:])

    def _apply_scalars(self, scalars):
        if not math.isfinite(scalars.projection):
            raise ValueError("detector returned a non-finite projection")
        self.gate_strength = scalars.gate_strength
        self.last_projection = scalars.projection

    def _trace_step(self, plan, step_t):
        """Queue this step's trace row; its numbers stay on the GPU until
        result_fields reads the whole response back at once."""
        self._trace_raw.append((
            {"step": self.step, "schedule_scale": self._schedule_scale,
             "source": plan.source, "num_generated_tokens": plan.n_committed,
             "phase": "block_recovery" if self.in_recovery else "base",
             "steer_armed": plan.steer},
            self._pending.fired, self.gate_strength, step_t))

    def _materialize_trace(self):
        """Turn queued trace rows into dicts with a single GPU read."""
        if not self._trace_raw:
            return
        values = torch.stack([t for *_, t in self._trace_raw]).tolist()
        for (fields, fired, latched, _), (proj, gate, alpha) in zip(
                self._trace_raw, values):
            if fired:
                if not math.isfinite(proj):
                    raise ValueError("detector returned a non-finite projection")
                self.gate_strength, self.last_projection = gate, proj
            self.trace.append({
                "step": fields["step"],
                "projection": proj if fired else None,
                "strength": gate if fired else latched,
                "schedule_scale": fields["schedule_scale"],
                "effective_alpha": alpha,
                "source": fields["source"],
                "num_generated_tokens": fields["num_generated_tokens"],
                "phase": fields["phase"],
                "steer_armed": fields["steer_armed"],
            })
        self._trace_raw = []

    @torch.no_grad()
    def forward(self, x, region, *, schedule_scale=1.0, logit_positions=None,
                n_masks=None):
        """One model forward with in-forward detection and steering.

        Detection, steering and (for V3) the deferred boundary audit all ride
        the same model call: _plan_forward decides what to do, the hooks do it
        during the call, and _read_scalars syncs the results back in one go.

        logit_positions: index tensor of positions the caller will read logits
        for (the eligible set). Given, ln_f's output is sliced to those rows so
        the vocab projection runs on n positions instead of the full sequence;
        output.logits is then [1, n, vocab] aligned to logit_positions."""
        self._schedule_scale = float(schedule_scale)
        plan = self._plan_forward(x, region, n_masks)
        self._pending = _GatePass(pool=plan.gate_pool)
        output, feats = self._run_hooked_forward(x, plan, logit_positions)
        step_t = self._step_tensor(x)

        # Python only needs these numbers when it has to branch on them: an
        # audit decides whether to recover, a step-0 binary gate latches the
        # strength later steps read. Any
        # other step leaves them on the GPU, so it never waits on the device.
        scalars = None
        if plan.audit is not None or (self.gate_once and self.step == 0):
            scalars = self._read_scalars(plan, feats, step_t)
            if self._pending.fired:
                self._apply_scalars(scalars)

        if plan.audit is not None and self._apply_audit(
                x, region,
                self._reading(scalars.audit_values, len(plan.audit.chunks)),
                audit=plan.audit):
            # The forward just consumed is stale post-recovery; redo it so the
            # sampler commits from post-recovery logits.
            self._pending = None
            return self.forward(x, region, schedule_scale=schedule_scale,
                                logit_positions=logit_positions,
                                n_masks=n_masks)

        self._trace_step(plan, step_t)
        self._pending = None
        self.step += 1
        return output

    def _remasked(self):
        return False

    def result_fields(self):
        self._materialize_trace()
        if not self.trace:
            return {}
        return {"gate_projection": self.trace[0]["projection"],
                "gate_open": any(t["strength"] > 0 for t in self.trace),
                "gate_max_strength": max(t["strength"] for t in self.trace),
                "remasked": self._remasked(),
                "gate_trace": self.trace}

    @staticmethod
    def summarize(results):
        graded = [r for r in results if "gate_open" in r]
        n_open = sum(r["gate_open"] for r in graded)
        n_remask = sum(r["remasked"] for r in graded)
        return (f"gate opened on {n_open}/{len(graded)} prompts, "
                f"remasked {n_remask}")

    def describe(self):
        return {"defense": self.name, "steer": self.steer_mode,
                "transform": self.transform, "strength": self.strength,
                "threshold": self.threshold, "width": self.width,
                "layers": self.steer_layers, "steer_shift": self.steer_shift, "remask": "none"}
