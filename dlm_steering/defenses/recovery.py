"""V3 response auditing, remasking, and block recovery."""
import math
from dataclasses import asdict
import torch
from dlm_steering.runtime.constants import MODEL_LOCK, TURN_BREAKERS
from sampler import take_true
from .base import _BoundaryReading, _GateReached, _PendingAudit, _prompt_text_mask
from .steering import Ours


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
                 recovery_alpha_growth=1.0, remask_prompt=False,
                 remask_prompt_tail=0, **kw):
        if recovery_steps != "auto" and recovery_steps <= 0:
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
        if remask_prompt_tail < 0:
            raise ValueError("remask_prompt_tail must be >= 0")
        self.remask_prompt_tail = int(remask_prompt_tail)
        self._prompt_text_slots = None
        self.recovery_steps = "auto" if recovery_steps == "auto" else int(recovery_steps)
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
        # A detector fitted with --prompt-tail N also pools the last N prompt
        # text tokens, so the text next to the answer (a DIJA template, an
        # injected prefix) is read together with it.
        self._det_prompt_tail = int(response_detector.get("prompt_tail", 0))
        self._det_tail_positions = None

    def defend(self, model, prompt_ids, rng=None, **gen_config):
        if self.remask_prompt or self._det_prompt_tail:
            self._prompt_text_slots = _prompt_text_mask(self.tokenizer, prompt_ids)
        if self._det_prompt_tail:
            # Text only: DIJA answer slots inside the prompt are already part
            # of the committed pool and would otherwise count twice.
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
        pools = self._audit_pools(
            (region[0] & (x[0] != self.mask_id)).nonzero().flatten(), chunks)
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
                    rng=None, sampling=None):
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
                              rng=rng, sampling=sampling)
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
            span_slots[:prompt_length] |= self._prompt_slots_to_reopen()
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
        recovery_steps = positions.numel() if self.recovery_steps == "auto" else self.recovery_steps
        recovery_steps = max(1, recovery_steps)
        event["extra_sampling_steps"] = recovery_steps
        counts = transfer_counts(positions.numel(), recovery_steps)
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
                "round": round_i,
                "steer_boost": self._steer_boost,
                "selected": positions,
                "num_span_positions": num_span_positions,
                "old_token_ids": old_tokens,
                "new_token_ids": x[0, positions].clone(),
                "round_sampling_forwards": recovery_steps,
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

    def _audit_pools(self, committed, chunks):
        tail = self._det_tail_positions
        if tail is None:
            return [committed, *chunks]
        return [torch.cat([tail, committed]), *chunks]

    def _prompt_slots_to_reopen(self):
        """Prompt text slots recovery reopens.

        Reopening every slot leaves the model nothing but the chat headers to
        condition on, so it fills prompt and answer alike with end-of-text and
        the response comes back empty. --remask-prompt-tail N reopens only the
        last N text slots -- the block right before the response, where DIJA
        templates and injected prefixes sit -- and keeps the request as context.
        """
        text = self._prompt_text_slots
        if not self.remask_prompt_tail:
            return text
        reopen = torch.zeros_like(text)
        reopen[text.nonzero().flatten()[-self.remask_prompt_tail:]] = True
        return reopen

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
        d.update(remask="v3", remask_prompt=self.remask_prompt,
                 remask_prompt_tail=self.remask_prompt_tail,
                 recovery_steps=self.recovery_steps,
                 recovery_rounds=self.recovery_rounds,
                 audit_all_boundaries=self.audit_all_boundaries,
                 audit_boundary=self.audit_boundary,
                 infill_checkpoint=self.infill_checkpoint,
                 recovery_alpha_growth=self.recovery_alpha_growth,
                 response_threshold=self._det_threshold,
                 response_detector_source=self.response_detector.get("source"),
                 response_detector_prompt_tail=self._det_prompt_tail,
                 response_detector_fingerprint=self.detector_fingerprint())
        return d

    def detector_fingerprint(self):
        """Short digest of the response detector's parameters.

        The boundary audit's probabilities are only comparable across runs that
        used the same weights, so threshold tuning must be able to tell two
        checkpoints apart even when they share a `source` string.
        """
        import hashlib
        digest = hashlib.sha256(self._det_weight.detach().cpu().numpy().tobytes())
        digest.update(repr(round(self._det_bias, 12)).encode())
        return digest.hexdigest()[:12]
