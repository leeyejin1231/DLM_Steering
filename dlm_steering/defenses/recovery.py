import hashlib
from dataclasses import asdict
import torch
from dlm_steering.runtime.constants import MODEL_LOCK, TURN_BREAKERS, block_index
from dlm_steering.runtime.models import model_blocks
from sampler import take_true
from .base import _BoundaryReading, _GateReached, _PendingAudit, _prompt_text_mask
from .steering import Ours


class V3(Ours):
    name = "v3"

    def __init__(self, model, *, response_detector, recovery_steps=32,
                 audit_all_boundaries=False, audit_boundary=0, infill_checkpoint=0,
                 remask_prompt=False, remask_prompt_frac=1.0, **kw):
        if recovery_steps != "auto" and recovery_steps <= 0:
            raise ValueError("recovery_steps must be positive")
        if audit_boundary < 0:
            raise ValueError("audit_boundary must be >= 0")
        if infill_checkpoint < 0:
            raise ValueError("infill_checkpoint must be >= 0")
        if response_detector is None:
            raise ValueError("--remask v3 requires a response detector checkpoint (--response-detector)")
        missing = {"weight", "bias", "threshold", "layer"} - response_detector.keys()
        if missing:
            raise ValueError(f"response detector missing keys: {sorted(missing)}")
        self.remask_prompt = remask_prompt
        if not 0.0 < remask_prompt_frac <= 1.0:
            raise ValueError("remask_prompt_frac must lie in (0, 1]")
        self.remask_prompt_frac = float(remask_prompt_frac)
        self._prompt_text_slots = None
        self.recovery_steps = "auto" if recovery_steps == "auto" else int(recovery_steps)
        self.audit_all_boundaries = bool(audit_all_boundaries)
        self.audit_boundary = int(audit_boundary)
        self.infill_checkpoint = int(infill_checkpoint)
        super().__init__(model, remask_enabled=True, **kw)
        blocks = model_blocks(model)
        self.det_layer = int(response_detector["layer"])
        if not 1 <= self.det_layer <= len(blocks):
            raise ValueError(f"response detector layer must be in 1..{len(blocks)}")
        self.det_block = blocks[block_index(self.det_layer)]
        self._det_pool_region = "region" in str(response_detector.get("pool", ""))
        self._audit_piggyback = (self.det_layer == self.gate_layer and not self._det_pool_region)
        device = self.gate_vector.device
        self._det_weight = torch.as_tensor(response_detector["weight"], dtype=torch.float32, device=device)
        self._det_bias = float(response_detector["bias"])
        self._det_threshold = float(response_detector["threshold"])
        self.response_detector = response_detector
        self._det_prompt_tail = int(response_detector.get("prompt_tail", 0))
        self._det_tail_positions = None

    def defend(self, model, prompt_ids, rng=None, **gen_config):
        if self.remask_prompt or self._det_prompt_tail:
            self._prompt_text_slots = _prompt_text_mask(self.tokenizer, prompt_ids)
        if self._det_prompt_tail:
            text = self._prompt_text_slots & (prompt_ids[0] != self.mask_id)
            self._det_tail_positions = text.nonzero().flatten()[-self._det_prompt_tail:]
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

    def _audit_vector(self, feats, det_feats=None):
        det = feats if det_feats is None else det_feats
        logit = det[0] @ self._det_weight + self._det_bias
        return torch.cat([feats @ self.gate_vector, logit.reshape(1), torch.sigmoid(logit.reshape(1))])

    def _reading(self, values, n_chunks):
        projection = values[0]
        return _BoundaryReading(
            projection=projection,
            block_projections=values[1:1 + n_chunks],
            response_logit=values[-2],
            response_probability=values[-1],
            strength=min(1.0, max(0.0, (projection - self.threshold) / self.width)))

    @torch.no_grad()
    def _audit(self, x, region, chunks):
        committed = (region[0] & (x[0] != self.mask_id)).nonzero().flatten()
        pools = self._audit_pools(committed, chunks)
        gate_out, det_out = [], []
        hooks = [(self.gate_block, self._audit_capture_hook(pools, gate_out))]
        if not self._audit_piggyback:
            det_pools = ([region[0].nonzero().flatten(), *chunks] if self._det_pool_region else pools)
            hooks.append((self.det_block, self._audit_capture_hook(det_pools, det_out)))
        stop_block = (self.det_block if self.det_layer > self.gate_layer else self.gate_block)

        def stop(module, inputs, output):
            raise _GateReached

        with MODEL_LOCK:
            handles = [block.register_forward_hook(hook) for block, hook in hooks]
            handles.append(stop_block.register_forward_hook(stop))
            try:
                self.model(x)
            except _GateReached:
                pass
            finally:
                for handle in handles:
                    handle.remove()
        self.audit_forwards += 1
        if len(gate_out) != 1 or len(det_out) != len(hooks) - 1:
            raise RuntimeError("audit hook must execute exactly once per forward")
        det = det_out[0] if det_out else None
        return self._reading(self._audit_vector(gate_out[0], det).tolist(), len(chunks))

    @torch.no_grad()
    def after_block(self, x, region, *, block_number, block_positions,
                    prompt_length, temperature, remasking, last_block=False,
                    rng=None, sampling=None):
        if self._pending_audit is not None:
            pending = self._pending_audit
            self._pending_audit = None
            reading = self._audit(x, region, self._chunk_positions(
                pending.block_row.nonzero().flatten()))
            self._apply_audit(x, region, reading, audit=pending)
        audit = _PendingAudit(block_number=block_number,
                              block_row=block_positions[0],
                              prompt_length=prompt_length,
                              temperature=temperature, remasking=remasking,
                              rng=rng, sampling=sampling)
        if last_block or not self._audit_piggyback:
            reading = self._audit(x, region, self._chunk_positions(
                block_positions[0].nonzero().flatten()))
            self._apply_audit(x, region, reading, audit=audit)
        else:
            self._pending_audit = audit

    def _apply_audit(self, x, region, reading, *, audit):
        """Record the audit; run recovery when the block still reads as a
        response. Returns True when recovery ran."""
        block_number = audit.block_number
        trigger = ((self.audit_all_boundaries or block_number == self.audit_boundary) and reading.response_probability >= self._det_threshold)
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

        event = {"boundary": block_number,
                 "pre_audit": asdict(reading),
                 "pre_recovery_token_ids": x[0].clone(),
                 "rounds": [], "applied": True,
                 "extra_sampling_steps": self.recovery_steps}
        self.recovery_events.append(event)

        prompt_length, temperature = audit.prompt_length, audit.temperature
        remasking = audit.remasking
        span_slots = region[0] & (x[0] != self.mask_id)
        span_slots[prompt_length:] = False
        if self.remask_prompt:
            span_slots[:prompt_length] |= self._prompt_slots_to_reopen(audit.rng)
        targets = audit.block_row | span_slots
        original_region_count = self._n_region
        region = region | targets[None]
        target_count, outside, self._n_region = torch.stack([
            targets.sum(), ((x[0] == self.mask_id) & region[0] & ~targets).sum(),
            region.sum()]).tolist()
        positions = take_true(targets, target_count)
        from sampler import commit_sample, transfer_counts
        recovery_steps = positions.numel() if self.recovery_steps == "auto" else self.recovery_steps
        recovery_steps = max(1, recovery_steps)
        event["extra_sampling_steps"] = recovery_steps
        counts = transfer_counts(positions.numel(), recovery_steps)
        num_span_positions = span_slots.sum()
        old_tokens = x[0, positions].clone()
        x.copy_(torch.where(targets[None], self.mask_id, x))
        eligible = positions
        self.in_recovery = True
        try:
            if audit.sampling and audit.sampling["decoder"] == "dream":
                from dream_sampler import dream_denoise
                rule = {k: v for k, v in audit.sampling.items() if k != "decoder"}
                def recovery_logits(cur):
                    logits = self.forward(cur, region, schedule_scale=1.0).logits
                    if self.remask_prompt:
                        user_positions = self._prompt_text_slots.nonzero().flatten()
                        banned = torch.tensor((*TURN_BREAKERS, self.mask_id), device=x.device)
                        logits[0, user_positions[:, None], banned[None, :]] = torch.finfo(logits.dtype).min
                    return logits
                dream_denoise(x, targets[None], recovery_steps,
                              recovery_logits, rng=audit.rng, **rule)
            else:
                for i in range(recovery_steps):
                    final = i == recovery_steps - 1
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
            "selected": positions,
            "num_span_positions": num_span_positions,
            "old_token_ids": old_tokens,
            "new_token_ids": x[0, positions].clone(),
            "round_sampling_forwards": recovery_steps,
            "post_trial_token_ids": x[0].clone()})
        self._n_region = original_region_count
        return True

    def _audit_pools(self, committed, chunks):
        tail = self._det_tail_positions
        if tail is None:
            return [committed, *chunks]
        return [torch.cat([tail, committed]), *chunks]

    def _prompt_slots_to_reopen(self, rng=None):
        text = self._prompt_text_slots
        if self.remask_prompt_frac >= 1.0:
            return text
        idx = text.nonzero().flatten()
        keys = torch.rand(idx.numel(), dtype=torch.float64, device=idx.device, generator=rng)
        chosen = idx[keys.argsort()[:round(self.remask_prompt_frac * idx.numel())]]
        reopen = torch.zeros_like(text)
        reopen[chosen] = True
        return reopen

    def _remasked(self):
        return any(e["applied"] for e in self.recovery_events)

    def result_fields(self):
        fields = super().result_fields()
        if fields and self.boundary_audits:
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
        d.update(remask="v3", remask_prompt=self.remask_prompt,
                 remask_prompt_frac=self.remask_prompt_frac,
                 recovery_steps=self.recovery_steps,
                 audit_all_boundaries=self.audit_all_boundaries,
                 audit_boundary=self.audit_boundary,
                 infill_checkpoint=self.infill_checkpoint,
                 response_threshold=self._det_threshold,
                 response_detector_layer=self.det_layer,
                 response_detector_pool=self.response_detector.get("pool"),
                 response_detector_source=self.response_detector.get("source"),
                 response_detector_prompt_tail=self._det_prompt_tail,
                 response_detector_fingerprint=self.detector_fingerprint())
        return d

    def detector_fingerprint(self):
        digest = hashlib.sha256(self._det_weight.detach().cpu().numpy().tobytes())
        digest.update(repr(round(self._det_bias, 12)).encode())
        return digest.hexdigest()[:12]
