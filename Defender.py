"""Defenses: policy objects the diffusion sampler consults every step.

exp.py selects one by name from DEFENDERS. The whole defended generation runs
through Defender.defend(); internally sampler.generate calls the policy
protocol once per denoising step:

    reset()          per response
    forward()        detection + steering inside a single model call
    after_block()    block-boundary hook: audit, remask, or budget extra steps
    result_fields()  per-response record fields

transform_prompt() is a separate prompt-side hook for defenses that only
edit the user message.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from common import MASK_ID, MODEL_LOCK, block_index, load_detector
from sampler import take_true


def _prompt_text_mask(tokenizer, prompt_ids):
    """Editable user text; retain the chat headers even when HF omits their
    IDs from all_special_ids (as LLaDA's tokenizer does)."""
    marker = "__PROMPT_CONTENT_BOUNDARY__"
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], tokenize=False,
        add_generation_prompt=True)
    prefix, suffix = template.split(marker)
    prefix_len = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    suffix_len = len(tokenizer(suffix, add_special_tokens=False).input_ids)
    special = torch.tensor(tokenizer.all_special_ids, device=prompt_ids.device)
    editable = ~torch.isin(prompt_ids[0], special)
    editable[:prefix_len] = False
    if suffix_len:
        editable[-suffix_len:] = False
    return editable


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace(output, hidden):
    return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


class _GateReached(Exception):
    """Stops a detector-only forward once the gate block has been captured.

    Nothing after blocks[gate_layer - 1] can change what the detector reads, so
    on a probe or audit pass the remaining blocks and the vocab projection are
    pure waste (~1.8x on this model, where the gate sits at layer 18 of 32).
    The captured features are bit-identical to a full forward's.
    """


# ---------------------------------------------------------------------------
# Per-forward state. Forward hooks cannot return values to their caller, so a
# forward() and the hooks it registers communicate through these objects; they
# live for exactly one model call.
# ---------------------------------------------------------------------------

@dataclass
class _GatePass:
    """Scratch shared by one forward() and its gate/steering hooks."""
    pool: Any = None        # positions the gate projection is averaged over
    fired: bool = False     # the gate hook ran (it is skipped when idle)
    projection: Any = None  # 0-d tensor; stays unsynced until _read_scalars
    alpha: Any = 0.0        # effective steering alpha the steer hook applied


@dataclass
class _ForwardPlan:
    """What one forward() reads, steers and audits -- decided before the call.

    Every index tensor is resolved here rather than inside a hook: a nonzero()
    in the middle of the model call would sync the GPU mid-forward.
    """
    source: str             # "generated" once any answer token is committed
    n_committed: int
    read_gate: bool
    steer: bool
    gate_pool: Any = None        # index tensor for the gate, or None
    steer_mask: Any = None       # bool [seq]: masked slots the steering hook pushes
    audit: Any = None            # _PendingAudit riding this forward, or None
    audit_pools: Any = None      # index tensors the audit hook pools over


@dataclass
class _StepScalars:
    """The scalars one forward needs, after a single batched GPU sync."""
    projection: float
    gate_strength: float
    effective_alpha: float
    audit_values: list = field(default_factory=list)


@dataclass
class _PendingAudit:
    """A finished block whose boundary audit rides the next forward.

    V3 defers non-final boundaries so the gate-layer features come from a
    forward the sampler needs anyway, saving one full pass per clean boundary.
    """
    block_number: int       # which block just finished (0-based)
    block_row: Any          # bool [seq]: that block's answer slots
    prompt_length: int
    temperature: float
    remasking: str
    rng: Any = None
    chunks: list = field(default_factory=list)


@dataclass
class _BoundaryReading:
    """What the response detector saw at one block boundary."""
    projection: float        # gate direction over every committed token
    block_projections: list  # ... and over each chunk of the finished block
    response_logit: float
    response_probability: float
    strength: float          # gate strength implied by `projection`


class Defender(ABC):
    """Per-run defense policy; the sampler owns token sampling."""

    name: str
    # gen_length 0 only: ask the sampler for an after_block audit every this
    # many committed mask slots (0 = only at the end). See sampler.generate.
    infill_checkpoint = 0

    @classmethod
    def add_args(cls, parser):
        """Register defense-specific CLI arguments (optional)."""

    @classmethod
    @abstractmethod
    def from_args(cls, args, model):
        """Build the policy: load bundles, construct, return the instance."""

    def prepare(self, tokenizer, vanilla_ids):
        """Provide token metadata and the clean reference for this row."""
        self.tokenizer = tokenizer
        self.vanilla_ids = vanilla_ids

    def transform_prompt(self, user_message):
        """Prompt-side hook; identity for activation-level defenses."""
        return user_message

    @abstractmethod
    def reset(self):
        """Reset per-response state."""

    @abstractmethod
    def forward(self, x, region, *, schedule_scale, logit_positions=None,
                n_masks=None):
        """One model forward, optionally with in-forward detection + steering.
        Must return the model output (with .logits). logit_positions, when
        given, restricts the vocab projection to those rows -- .logits is then
        aligned to them. Implementations may ignore it (full logits).
        n_masks, when given, is the number of still-masked region slots, which
        the sampler knows without asking the GPU."""

    def after_block(self, x, region, *, block_number, block_positions,
                    prompt_length, temperature, remasking, last_block=False,
                    rng=None):
        """Called once after each block's denoising loop completes.

        block_number: which block just finished, 0-based (not to be confused
        with common.block_index, which maps a layer to a transformer block).
        block_positions: answer slots of the just-finished block. prompt_length
        marks the prompt/generation boundary; committed answer slots before it
        (e.g. filled DIJA spans) are also remaskable. last_block marks the
        final block (no later forward exists to piggyback an audit on). May
        audit the block or rewrite committed tokens back to MASK_ID. rng is
        the row's torch.Generator, needed by implementations that resample
        (V3 recovery).
        """

    @abstractmethod
    def result_fields(self):
        """Fields merged into this response's result record."""

    @staticmethod
    def summarize(results):
        """One-line run summary of a finished run, or None.

        Static like Evaluator.summarize so the --gpus parent can summarise the
        merged shards without building a policy of its own.
        """
        return None

    def describe(self):
        """Config dict stored in the result payload."""
        return {"defense": self.name}

    def defend(self, model, prompt_ids, rng=None, **gen_config):
        """Run one defended generation through the unified sampler."""
        from sampler import generate
        return generate(model, prompt_ids, self, rng=rng, **gen_config)

    def defend_batch(self, model, prompt_ids_list, rng=None, **gen_config):
        """A batch of defended generations; sequential by default.

        Stateful defenses (remask/recovery branch per sequence) cannot share
        a denoising loop, so only NullDefender overrides this with the truly
        batched sampler.
        """
        return [self.defend(model, ids, rng=rng, **gen_config)
                for ids in prompt_ids_list]


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

    def forward(self, x, region, *, schedule_scale, logit_positions=None,
                n_masks=None):
        if logit_positions is None:
            return self.model(x)
        ln_f = self.model.model.transformer.ln_f
        with MODEL_LOCK:
            handle = ln_f.register_forward_hook(
                lambda m, i, o: o[:, logit_positions])
            try:
                return self.model(x)
            finally:
                handle.remove()

    def defend_batch(self, model, prompt_ids_list, rng=None, **gen_config):
        """No hooks -> rows denoise uniformly; safe to share one loop."""
        from sampler import generate_batch
        return generate_batch(model, prompt_ids_list, rng=rng, **gen_config)

    def result_fields(self):
        return {}

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
            self.sites.append((blocks[block_index(layer)],
                               vector.float() / vector.norm(), float(ref_norm)))
            self.steer_layers.append(layer)
        if self.steer_enabled and not self.sites:
            raise ValueError("at least one steering site is required when steering is on")
        self.model = model
        self.gate_layer = gate_layer
        self.gate_block = blocks[block_index(gate_layer)]
        self.ln_f = model.model.transformer.ln_f
        self.gate_vector = gate_vector.float()
        self.threshold, self.width = float(threshold), float(width)
        self.strength, self.transform = float(strength), transform
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
        parser.add_argument("--response-detector", default="outputs/response_detector.pt",
                            help="Logistic-regression response checkpoint; required by "
                                 "--remask v3*. Its layer must match --detector-layer.")
        parser.add_argument("--remask-prompt", action="store_true",
                            help="v3: recover all prompt text, preserving special tokens.")
        parser.add_argument("--recovery-steps", type=int, default=32,
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
            for layer in (int(s) for s in args.layer.split(",")):
                li = bundle["layers"].index(layer)
                sites.append((layer, bundle["vector"][li].to(device), bundle["mean_act_norm"][li]))
        det_vec, det_layer, threshold = load_detector(
            args.detector, args.detector_layer, device, args.gate_threshold)
        shared = dict(model=model, gate_layer=det_layer, gate_vector=det_vec,
                      threshold=threshold, width=args.gate_width, sites=sites,
                      strength=args.alpha, transform=args.transform,
                      steer=args.steer)
        if args.remask == "none":
            return cls(**shared)
        response_detector = torch.load(args.response_detector, map_location="cpu",
                                       weights_only=False)
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

        return _ForwardPlan(
            source="generated" if n_committed else "masked",
            n_committed=n_committed, read_gate=read_gate, steer=steer,
            gate_pool=gate_pool,
            steer_mask=masks[0] if steer else None,
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
                if logit_positions is not None:
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
                "layers": self.steer_layers, "remask": "none"}


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
                 audit_boundary=0, infill_checkpoint=0,
                 recovery_alpha_growth=1.0, remask_prompt=False, **kw):
        if recovery_steps <= 0:
            raise ValueError("recovery_steps must be positive")
        if recovery_rounds <= 0:
            raise ValueError("recovery_rounds must be positive")
        if audit_boundary < 0:
            raise ValueError("audit_boundary must be >= 0")
        if infill_checkpoint < 0:
            raise ValueError("infill_checkpoint must be >= 0")
        if not math.isfinite(recovery_alpha_growth) or recovery_alpha_growth <= 0:
            raise ValueError("recovery_alpha_growth must be finite and positive")
        if response_detector is None:
            raise ValueError("--remask v3 requires a response detector "
                             "checkpoint (--response-detector)")
        missing = {"weight", "bias", "threshold", "layer"} - response_detector.keys()
        if missing:
            raise ValueError(f"response detector missing keys: {sorted(missing)}")
        self.remask_prompt = remask_prompt
        self._prompt_text_slots = None
        self.recovery_steps = int(recovery_steps)
        self.recovery_rounds = int(recovery_rounds)
        self.audit_all_boundaries = bool(audit_all_boundaries)
        self.audit_boundary = int(audit_boundary)
        self.infill_checkpoint = int(infill_checkpoint)
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

    def defend(self, model, prompt_ids, rng=None, **gen_config):
        if self.remask_prompt:
            self._prompt_text_slots = _prompt_text_mask(self.tokenizer, prompt_ids)
        return super().defend(model, prompt_ids, rng=rng, **gen_config)

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
        """Parse the audit slice of a forward's scalars into a named reading.

        `values` is the tail _audit_vector produced: the committed-pool
        projection, one projection per block chunk, then the response
        detector's logit and probability.
        """
        projection = values[0]
        return _BoundaryReading(
            projection=projection,
            block_projections=values[1:1 + n_chunks],
            response_logit=values[-2],
            response_probability=values[-1],
            strength=min(1.0, max(0.0,
                                  (projection - self.threshold) / self.width)))

    @torch.no_grad()
    def _audit(self, x, region, chunks):
        """One unsteered forward capturing gate-layer features pooled over the
        committed tokens and each chunk of the finished block."""
        pools = [(region[0] & (x[0] != self.mask_id)).nonzero().flatten(), *chunks]
        feats_out = []
        capture = self._audit_capture_hook(pools, feats_out)

        def capture_and_stop(module, inputs, output):
            capture(module, inputs, output)
            raise _GateReached

        # Audits read only gate-layer features, so the forward stops there --
        # see _GateReached. ln_f is never reached, vocab projection included.
        with MODEL_LOCK:
            handle = self.gate_block.register_forward_hook(capture_and_stop)
            try:
                self.model(x)
            except _GateReached:
                pass
            finally:
                handle.remove()
        self.audit_forwards += 1
        if len(feats_out) != 1:
            raise RuntimeError("audit hook must execute exactly once per forward")
        return self._reading(self._audit_vector(feats_out[0]).tolist(), len(chunks))

    @torch.no_grad()
    def after_block(self, x, region, *, block_number, block_positions,
                    prompt_length, temperature, remasking, last_block=False,
                    rng=None):
        if self._pending_audit is not None:
            # A deferred audit whose next forward never arrived still runs.
            pending = self._pending_audit
            self._pending_audit = None
            reading = self._audit(x, region, self._chunk_positions(
                pending.block_row.nonzero().flatten()))
            self._apply_audit(x, region, reading, audit=pending)
        audit = _PendingAudit(block_number=block_number,
                              block_row=block_positions[0],
                              prompt_length=prompt_length,
                              temperature=temperature, remasking=remasking,
                              rng=rng)
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
        block_number = audit.block_number
        trigger = ((self.audit_all_boundaries
                    or block_number == self.audit_boundary)
                   and reading.response_probability >= self._det_threshold)
        if self.audit_all_boundaries:
            rule = "response_probability_cutoff_each_boundary"
        elif self.audit_boundary == 0:
            rule = "response_probability_cutoff_first_boundary"
        else:
            rule = f"response_probability_cutoff_boundary_{self.audit_boundary}"
        self.boundary_audits.append({
            "boundary": block_number, **asdict(reading), "trigger": trigger,
            "trigger_rule": rule})
        if not trigger:
            return False
        self.triggered = True

        event = {"boundary": block_number,
                 "pre_audit": asdict(reading),
                 "pre_recovery_token_ids": x[0].clone(),
                 "rounds": [], "applied": True,
                 "extra_sampling_steps": self.recovery_steps}
        self.recovery_events.append(event)

        # Remask the finished block plus any committed answer slots inside the
        # prompt (DIJA spans carry the payload under that attack). With
        # recovery_rounds > 1 the block is re-audited after each regeneration
        # and remasked again while it still reads as a response.
        prompt_length, temperature = audit.prompt_length, audit.temperature
        remasking = audit.remasking
        span_slots = region[0] & (x[0] != self.mask_id)
        span_slots[prompt_length:] = False
        if self.remask_prompt:
            span_slots[:prompt_length] |= self._prompt_text_slots
        targets = audit.block_row | span_slots
        # Recovery may reopen fixed prompt text. Include it in recovery pools
        # only; subsequent ordinary audits still use the original answer region.
        original_region_count = self._n_region
        region = region | targets[None]
        # Read all counts together; avoid an extra synchronization for the
        # temporary recovery region and nonzero's separate size sync.
        target_count, outside, self._n_region = torch.stack([
            targets.sum(), ((x[0] == self.mask_id) & region[0] & ~targets).sum(),
            region.sum()]).tolist()
        positions = take_true(targets, target_count)
        from sampler import commit_sample, transfer_counts
        counts = transfer_counts(positions.numel(), self.recovery_steps)
        num_span_positions = span_slots.sum()
        for round_i in range(self.recovery_rounds):
            # Re-detected rounds steer harder: strength *= growth ** round_i.
            self._steer_boost = self.recovery_alpha_growth ** round_i
            old_tokens = x[0, positions].clone()
            # Reuse the initial count for round zero. Later rounds still read
            # it in case the model itself predicted MASK_ID during recovery.
            if round_i:
                outside = int(((x[0] == self.mask_id) & region[0] & ~targets).sum())
            x.copy_(torch.where(targets[None], self.mask_id, x))
            eligible = positions
            self.in_recovery = True
            try:
                for i in range(self.recovery_steps):
                    final = i == self.recovery_steps - 1
                    if eligible.numel() == 0:
                        break
                    if counts[i] == 0 and not final:
                        continue
                    logits = self.forward(x, region, schedule_scale=1.0,
                                          logit_positions=eligible,
                                          n_masks=outside + eligible.numel()).logits
                    eligible = commit_sample(x, logits, eligible, counts[i],
                                             temperature, remasking, final=final,
                                             rng=audit.rng)
            finally:
                self.in_recovery = False
            event["rounds"].append({
                "round": round_i,
                "steer_boost": self._steer_boost,
                "selected": positions,
                "num_span_positions": num_span_positions,
                "old_token_ids": old_tokens,
                "new_token_ids": x[0, positions].clone(),
                "round_sampling_forwards": self.recovery_steps,
                "post_trial_token_ids": x[0].clone()})
            if round_i + 1 >= self.recovery_rounds:
                break
            post = self._audit(x, region, self._chunk_positions(positions))
            self.boundary_audits.append({
                "boundary": block_number, **asdict(post), "trigger": False,
                "trigger_rule": "post_recovery_reaudit"})
            if post.response_probability < self._det_threshold:
                break
        self._steer_boost = 1.0
        self._n_region = original_region_count
        return True

    def _remasked(self):
        return any(e["applied"] for e in self.recovery_events)

    def result_fields(self):
        fields = super().result_fields()
        if fields and self.boundary_audits:
            # Recovery snapshots are diagnostic only. Transfer them together
            # after sampling, preserving scalar/list shapes in the JSON schema.
            tensors = []

            def collect(value):
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
                elif isinstance(value, dict):
                    for item in value.values():
                        collect(item)
                elif isinstance(value, list):
                    for item in value:
                        collect(item)

            collect(self.recovery_events)
            if tensors:
                values = iter(torch.cat([t.reshape(-1) for t in tensors]).tolist())

                def materialize(value):
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 0:
                            return next(values)
                        return [next(values) for _ in range(value.numel())]
                    if isinstance(value, dict):
                        return {k: materialize(v) for k, v in value.items()}
                    if isinstance(value, list):
                        return [materialize(v) for v in value]
                    return value

                self.recovery_events = materialize(self.recovery_events)
            fields.update(boundary_audits=self.boundary_audits,
                          recovery_events=self.recovery_events,
                          audit_forwards=self.audit_forwards)
        return fields

    def describe(self):
        d = super().describe()
        d.update(remask="v3", remask_prompt=self.remask_prompt, recovery_steps=self.recovery_steps,
                 recovery_rounds=self.recovery_rounds,
                 audit_all_boundaries=self.audit_all_boundaries,
                 audit_boundary=self.audit_boundary,
                 infill_checkpoint=self.infill_checkpoint,
                 recovery_alpha_growth=self.recovery_alpha_growth)
        return d


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


class DiffuGuard(NullDefender):
    """Adapter to the included author's LLaDA generator."""

    name = "diffuguard"
    needs_vanilla = True

    @classmethod
    def add_args(cls, parser):
        parser.set_defaults(remasking="adaptive_step")
        parser.add_argument("--sp-threshold", type=float, default=0.2)
        parser.add_argument("--refinement-steps", type=int, default=8)
        parser.add_argument("--remask-ratio", type=float, default=0.9)
        parser.add_argument("--repair-scope", choices=["all", "first"], default="all",
                            help="DiffuGuard repair eligibility: all answer blocks "
                                 "(default), or only the first block. Prompt text "
                                 "is eligible in the first block in both modes.")

    @classmethod
    def from_args(cls, args, model):
        if not math.isfinite(args.sp_threshold) or not 0 <= args.remask_ratio <= 1:
            raise ValueError("DiffuGuard needs a finite threshold and remask ratio in [0, 1]")
        if args.refinement_steps <= 0:
            raise ValueError("--refinement-steps must be positive")
        obj = cls(model)
        obj.options = dict(sp_threshold=args.sp_threshold,
                           refinement_steps=args.refinement_steps,
                           remask_ratio=args.remask_ratio,
                           correct_only_first_block=args.repair_scope == "first")
        if args.row_workers != 1:
            raise ValueError("DiffuGuard uses global RNG; use --row-workers 1")
        return obj

    @torch.no_grad()
    def defend(self, model, prompt_ids, rng=None, **gen_config):
        from third_party import diffuguard as backend
        generate = backend.generate
        if self.vanilla_ids is None:
            raise ValueError("DiffuGuard hidden detection requires a clean reference")
        length, block, steps = (gen_config[k] for k in ("gen_length", "block_length", "steps"))
        if length < 0 or block <= 0 or steps <= 0:
            raise ValueError("invalid DiffuGuard length/step settings")
        if length and (length % block or steps % (length // block)):
            raise ValueError("gen_length must divide into blocks and steps into block steps")
        if length > block and getattr(backend, "BLOCK_SCHEDULE_VERSION", None) != "prompt_then_answer_blocks_v1":
            raise RuntimeError("Incompatible bundled DiffuGuard block schedule")
        with MODEL_LOCK, torch.random.fork_rng(devices=[prompt_ids.device]):
            if rng is not None:
                # Advance the row generator across iterative attack candidates.
                # Reusing initial_seed() restarted the same stream each time.
                torch.cuda.set_rng_state(rng.get_state(), prompt_ids.device)
            baseline = model(self.vanilla_ids, output_hidden_states=True,
                             return_dict=True).hidden_states[-1].mean(dim=1).squeeze(0)
            protected = ~_prompt_text_mask(self.tokenizer, prompt_ids)[None]
            protected &= prompt_ids != MASK_ID
            output = generate(model, self.tokenizer, prompt_ids,
                            steps=gen_config["steps"], gen_length=gen_config["gen_length"],
                            block_length=gen_config["block_length"],
                            temperature=gen_config["temperature"],
                            remasking=gen_config["remasking"], cfg_scale=0.0,
                            sp_mode="hidden", baseline_hidden=baseline,
                            fill_all_masks=True, protected_index=protected,
                            attack_method="DIJA", **self.options)
            if rng is not None:
                rng.set_state(torch.cuda.get_rng_state(prompt_ids.device))
            return output

    def defend_batch(self, model, prompt_ids_list, rng=None, **gen_config):
        return Defender.defend_batch(self, model, prompt_ids_list, rng=rng, **gen_config)

    def describe(self):
        return {"defense": self.name, "implementation": "author generator",
                "sp_mode": "hidden", "fill_all_masks": True,
                "protect_special_tokens": True,
                "repair_scope": "first" if self.options["correct_only_first_block"] else "all",
                "block_schedule": "prompt_then_answer_blocks_v1", **self.options}


DEFENDERS = {d.name: d for d in (NullDefender, Ours, SelfReminder, DiffuGuard)}
