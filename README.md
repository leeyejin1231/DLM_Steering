# DLM_Steering

Gated activation steering with response-detector remasking (the `ours` defense) for
masked-diffusion language models, evaluated against jailbreak attacks (PAP, PAIR, DIJA)
next to the DiffuGuard and Self-Reminder baselines. Target models are
`GSAI-ML/LLaDA-8B-Instruct` (default) and `Dream-org/Dream-v0-Instruct-7B` (`--model dream`).

## 1. Environment and data

Run every command from the repository root.

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
# or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python data_downloader.py
```

`requirements.txt` is fully pinned (torch 2.3.1 with CUDA 12.1 wheels, transformers 4.55.4).
Model weights are fetched from the Hugging Face Hub on first use; point `HF_HOME` at an
existing cache if you keep one. Downloading AdvBench and HarmBench requires accepting the
dataset terms on the Hub and authenticating (`huggingface-cli login` or `HF_TOKEN`).

| Benchmark | `--source` | Rows |
|---|---|---:|
| JBB | `jbb_harmful` | 100 |
| HarmBench | `harmbench` | 393 |
| StrongREJECT | `strongreject` | 313 |

`--n` defaults to **20**; pass the row counts above for a full evaluation. PAP and PAIR use
the original datasets; only DIJA uses its own refined prompts.

Graders: attack success is judged by `meta-llama/Llama-Guard-4-12B` (gated on the Hub) and
by the StrongREJECT rubric with `gpt-oss:20b` served by Ollama. The evaluation scripts start
the Ollama container themselves; `ollama_setting/` holds the podman image and model pull
scripts.

`exp.py`, `pap_generate.py` and the graders show one tqdm bar per task; when a run is
sharded over several GPUs, each child process writes its full output to a log next to its
part file.

### Defenses

| `--defense` | Description | Requires |
|---|---|---|
| `none` | no defense | nothing |
| `ours` | gate, steering and V3 remasking | the checkpoints below |
| `diffuguard` | DiffuGuard | the generator code bundled in `third_party/` |
| `selfreminder` | a safety reminder prepended to the prompt | nothing |

The default paths of `ours` expect these files; they have to be prepared separately
(`script/build_vectors.sh` fits the vector, the gate detector and its threshold, and
`python -m dlm_steering.fitting.fit_response_detector` the response detector).

```text
outputs/steer_vector.pt
outputs/steer_detector.pt
outputs/gate_threshold.json
outputs/response_detector.pt
```

## 2. PAP: generate the attack, run the defenses, grade

### 2-1. Generate the attack prompts

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python pap_generate.py \
  --source jbb_harmful --seed 42 --reproduct \
  --out data/attacks/pap_better/jbb_harmful/seed42.json

# several GPUs: one worker per GPU, shards validated and merged into --out
.venv/bin/python pap_generate.py --source jbb_harmful --seed 42 --reproduct --gpus 0,1,2,3 \
  --out data/attacks/pap_better/jbb_harmful/seed42.json
```

- The default attacker model is **Qwen/Qwen3-14B** (change with `--model`).
- One attack sentence is generated per row. There are no retries and no internal grading.

### 2-2. Run two defenses on the same attack cache

The default cache path is picked automatically; `--pap-cache <file>` selects another one.

An attack cache is **generated once per dataset and seed** and shared by every model and
defense under comparison. This step does not load the attacker model.

```bash
# V3
CUDA_VISIBLE_DEVICES=0 .venv/bin/python exp.py \
  --attack pap \
  --defense ours --alpha 1 --steer adaptive --remask v3 --remask-prompt \
  --source jbb_harmful --n 100 --seed 42 --reproduct \
  --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 \
  --out outputs/JBB-pap-v3-42.json

# DiffuGuard
CUDA_VISIBLE_DEVICES=0 .venv/bin/python exp.py \
  --attack pap \
  --defense diffuguard \
  --source jbb_harmful --n 100 --seed 42 --reproduct \
  --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 \
  --out outputs/JBB-pap-diffuguard-42.json
```

`--source`, `--seed` and `--reproduct` must match the settings the cache was generated with.
For seeds 43 and 44 generate the cache of that seed first. Other datasets follow the same order.

To attack another target model directly, feed `results[*].attack_prompt` from the cache
JSON as the user message; `prompt` is the original benchmark question and `index` the
original row id. Caches live under `data/`.

## 3. Other attacks and options

### DIJA

DIJA's Qwen-refined prompts for the three datasets ship in `data/dija/`. They come from the
[DIJA repository](https://github.com/ZichenWen1/DIJA); the license is in `data/dija/LICENSE`.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python exp.py \
  --attack dija --defense diffuguard \
  --source jbb_harmful --n 100 --seed 42 --reproduct \
  --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 \
  --out outputs/JBB-dija-diffuguard-42.json
```

DIJA's `--gen-length` defaults to 128. Pass `--gen-length 0` to fill only the mask slots
inside the prompt.

### PAIR

PAIR refines its attack from the target's responses, so unlike PAP nothing is generated in
advance. The default attacker is Qwen/Qwen3-14B and the internal success check is the GCG
keyword test. The default search budget is **5 streams x 5 iterations**.

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python exp.py \
  --attack pair --defense diffuguard \
  --source jbb_harmful --n 100 --seed 42 --reproduct \
  --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 \
  --out outputs/JBB-pair-diffuguard-42.json
```

PAIR uses two GPUs: one for the target, one for the attacker model.

### Common options

- `--attack`: `none`, `prefix`, `dija`, `pap`, `pair`.
- `--seed` and `--reproduct`: random seed and deterministic kernels. Bit-identical results
  across different GPU architectures are not guaranteed.
- `--start`, `--n`: the row range of the source dataset to run.
- `--gpus 0,1,2,3`: shard the prompt set over several GPUs, one process per GPU over a
  contiguous slice, merged into `--out`. Rows are seeded by index, so the merged file is
  identical to a single-GPU run. PAIR takes two GPUs per shard. Use this instead of the
  `CUDA_VISIBLE_DEVICES=...` of the single-GPU examples.
- V3: `--remask-prompt` also remasks the prompt text (special tokens excluded) when the
  response detector triggers; `--remask-prompt-frac r` reopens a random fraction `r` of it.
- DiffuGuard defaults: SAR (`adaptive_step`), `repair-scope all`, threshold 0.2,
  8 refinement steps, remask ratio 0.9.

The attack- and defense-specific options are listed in the help of the selected pair.

```bash
.venv/bin/python exp.py --attack pap --defense ours --help
.venv/bin/python exp.py --attack pair --defense diffuguard --help
.venv/bin/python pap_generate.py --help
```

## 4. Evaluation

```bash
.venv/bin/python eval_llamaguard.py \
  --in outputs/JBB-pap-v3-42.json --out outputs/JBB-pap-v3-42_lg4.json --gpus 0

.venv/bin/python run_sr_eval.py \
  --in outputs/JBB-pap-v3-42.json --out outputs/JBB-pap-v3-42_sr.json --gpu 0 --port 50001

.venv/bin/python script/report.py \
  outputs/JBB-pap-v3-42_lg4.json outputs/JBB-pap-v3-42_sr.json
```

- **LG4**: attack success rate from Llama-Guard-4's unsafe verdicts.
- **GPT-OSS**: StrongREJECT scores from Ollama's `gpt-oss:20b`.

To grade on several GPUs with automatically managed local Ollama servers, pass
`--auto-server --gpus 0,1 --startup-workers 2` to `run_sr_eval.py`. Up to two servers are
initialised by default: CUDA discovery runs sequentially, model loading overlaps, and a
transient start-up failure is retried twice. `--startup-workers 1` serialises the model loads
too. `--workers` is the per-server grading concurrency once the servers are up.

### Over-refusal and utility

| Purpose | `--source` at generation | Evaluation |
|---|---|---|
| Over-refusal | `truthfulqa`, `xstest_safe`, ... | `python -m dlm_steering.fitting.judge_refusal --in <generations.json> --out <judged.json> --auto-server --gpus 0,1` |
| Utility | `mmlu`, `gsm8k`, `math500`, `truthfulqa_mc` | `python eval_utility.py --in <generations.json> --out <scored.json>` |

Fetch `gsm8k`, `math500` and `truthfulqa` into `data/` with
`python data_downloader.py gsm8k math500 truthfulqa`. For `math500` the last `\boxed{}` is
compared with the reference under the MATH repository's normalisation rules.

Generate these runs with `--attack none` and compare against `--defense none` over the same
row range. Use `.venv/bin/python` for the commands above as well.

## 5. Dream-v0-Instruct-7B

The same method (gated steering + V3 remasking) runs unchanged on
`Dream-org/Dream-v0-Instruct-7B`. With `--model dream` the registry in `models.py` supplies
the model constants (mask token `<|mask|>` = 151666, block path `model.model.layers`,
28 layers, `<|im_end|>`) and the default vector/detector paths move to `outputs/dream/`.
`--model` is read from the command line when `models.py` is imported, so it must be given there
(`--gpus` shard children receive it automatically). The environment is the same `.venv` as
for LLaDA.

### Deployed settings and checkpoints (`outputs/dream/`)

| Artifact | Layer | Notes |
|---|---|---|
| `steer_detector.pt` + `gate_threshold.json` | 14 (`best_layer`) | gate; mean OOD AUROC 0.970, threshold 2.706 |
| `steer_vector.pt` | **20** (pass `--layer 20`) | validation AUROC peaks at 17, but in the alpha sweep layer 17 collapses from alpha 1.5 while layer 20 refuses 100% at alpha 1 and stays fluent |
| `response_detector3_committed_L20.pt` | **20**, cutoff 0.387 | V3 response detector, fitted on first-block-boundary states (`dlm_steering/fitting/fit_boundary_detector.py`) |

Because the response detector's layer (20) differs from the gate layer (14), V3's boundary
audit does not ride the next defended forward: it runs a separate unsteered forward that
stops at the detector layer (`_audit_piggyback` in `dlm_steering/defenses/recovery.py`).
When the two layers coincide, as for LLaDA, the audit piggybacks as before. The checkpoint's
`layer` and `pool` keys decide this, and the result JSON records it under
`defense.response_detector_layer`.

```bash
# one run with the deployed settings (over-refusal, XSTest-safe, 250 prompts)
.venv/bin/python exp.py --model dream --attack none --defense ours --remask v3 --steer adaptive --layer 20 \
  --response-detector outputs/dream/response_detector3_committed_L20.pt \
  --remask-prompt --remask-prompt-frac 0.8 --source xstest_safe --n 250 \
  --gen-length 128 --steps 128 --block-length 32 --seed 42 --gpus 0,1 \
  --out outputs/dream/XSTest-safe-none-v3rp80-42.json
```

### Experiment scripts

The drivers of section 6 take `MODEL=dream`; the deployed Dream flags above come from
`ours_args dream` in `script/common.sh`. The runs behind the Dream results were tagged
`v3rp80`, so `TAG=v3rp80` resumes them:

```bash
MODEL=dream TAG=v3rp80 ATTACK=pap SOURCES=jbb_harmful script/run_benchmark.sh
MODEL=dream TAG=v3rp80 TQA_JUDGE=1 script/run_overrefusal.sh
MODEL=dream TAG=v3rp80 script/run_utility.sh
```

To refit the response detector (generate per arm, label with Llama Guard 4, refit on CPU):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dlm_steering.fitting.fit_boundary_detector --model dream --arms wildjailbreak,alpaca --gen-only
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m dlm_steering.fitting.fit_boundary_detector --model dream --arms wj_benign --gen-only
.venv/bin/python -m dlm_steering.fitting.fit_boundary_detector --model dream --pool committed --layer 20 --balanced \
  --from-samples outputs/dream/boundary_samples_{wildjailbreak,wj_benign,alpaca}.pt \
  --out outputs/dream/response_detector3_committed_L20.pt
```

The vector, the gate detector and its threshold are built with the same
`MODEL=dream script/build_vectors.sh` path as for LLaDA. On Dream, omitting
`--detector-layer` / `--layer` falls back to each bundle's `best_layer`, so pass `--layer 20`
for the vector explicitly.

## 6. Scripts

`script/` holds the end-to-end drivers. Each one sources `script/common.sh`, which picks the
interpreter (`PY`, default `.venv/bin/python`), the GPUs (`GPUS`, default: every visible card;
`PROCS_PER_GPU` workers per card) and the deployed `ours` flags per model (`ours_args`).
Every driver skips output files that are already complete, so an interrupted run resumes.

| Script | Runs | Main variables |
|---|---|---|
| `build_vectors.sh` | contrast pairs, steering vector, gate detector, gate threshold (`MODEL=dream` for Dream) | `--force` rebuilds |
| `run_benchmark.sh` | one attack under one defense on the harmful sets and seeds, then Llama Guard 4 + StrongREJECT + report (gen 128, 128 steps, block 32, temperature 0.2, `--reproduct`) | `MODEL`, `ATTACK` (pap/dija/pair), `DEFENSE` (ours/diffuguard/none/selfreminder), `SOURCES`, `SEEDS`, `TAG`, `DEFENSE_ARGS`, `GEN`/`STEPS`/`BLOCK`/`TEMPERATURE` |
| `run_overrefusal.sh` | benign sets under the defense, XSTest three-way refusal judge, per-seed summary `OR-summary-<TAG>.{json,md}`; `TQA_JUDGE=1` adds the TruthfulQA truthful/informative judge | `MODEL`, `DEFENSE`, `SOURCES` (xstest_safe truthfulqa), `SEEDS`, `TAG` |
| `run_utility.sh` | graded sets with and without the defense, scored by `eval_utility.py` | `MODEL`, `SOURCES` (math500 truthfulqa_mc, also mmlu gsm8k), `DEFS`, `SEEDS`, `TAG` |
| `run_ablation_dija_jbb.sh` | the four LLaDA ablation conditions on DIJA/JBB plus grading (`CONDS="allbnd bnd1 bnd2 bnd3"` for the boundary analysis) | `SEED` |
| `report.py` | one table over grader payloads (`*_lg4`, `*_sr`, `*_judged`, `*_acc`) | file arguments, or none for everything under `outputs/` |

```bash
# main table, LLaDA: PAP and DIJA under ours and DiffuGuard, three seeds each
for A in pap dija; do for D in ours diffuguard; do ATTACK=$A DEFENSE=$D script/run_benchmark.sh; done; done
# DiffuGuard in its authors' infilling setting (DIJA spans only, 200 steps)
ATTACK=dija DEFENSE=diffuguard TAG=diffuguard-infill GEN=0 STEPS=200 BLOCK=200 script/run_benchmark.sh
# over-refusal and utility for the deployed LLaDA defense
script/run_overrefusal.sh
script/run_utility.sh
```

The deployed LLaDA flags are steering layer 25, gate detector layer 18 with threshold 7.0,
the response detector `outputs/response_detector.pt` with cutoff 0.12, alpha 1, adaptive
steering and V3 remasking with an 80% random prompt remask; pass `DEFENSE_ARGS="..."` to run
anything else.

## Code layout

The attack, defense, evaluation and fitting code is organised by role under `dlm_steering/`;
the top-level scripts are the CLI entry points, the samplers and the model registry.

```text
dlm_steering/
├── paths.py          # repository and data paths
├── attacks/          # attack interface, prefix, DIJA, PAP, PAIR
├── defenses/         # defense interface, steering, V3 recovery, baselines
├── evaluation/       # graders, JSONL resume, per-attack aggregation, cache, result format
├── runtime/          # model loading, data, reproducibility, GPU sharding, JSON writing
└── fitting/          # vector / detector / threshold fitting and the refusal and TruthfulQA judges (python -m dlm_steering.fitting.<module>)
```

| File / directory | Role |
|---|---|
| `exp.py` / `experiment_row.py` | experiment runner / per-row generation and result record |
| `eval_llamaguard.py` / `run_sr_eval.py` | LG4 / GPT-OSS grading entry points |
| `pap_generate.py` / `pap_common.py` | PAP attack generation / technique assignment and cache validation |
| `sampler.py` / `dream_sampler.py` | LLaDA / Dream diffusion samplers |
| `models.py` / `model_loading.py` | target model registry / pretrained weight loading |
| `ollama_runtime.py` | start and stop of the dedicated Ollama servers |
| `dlm_steering/fitting/` | vector, detector and threshold fitting, the refusal and TruthfulQA judges and the over-refusal summary; run as `python -m dlm_steering.fitting.<module>` |
| `dlm_steering/fitting/fit_boundary_detector.py` / `judge_truthfulqa.py` / `aggregate_overrefusal.py` | first-boundary response detector (Dream) / TruthfulQA truthful-informative judge / over-refusal aggregation over seeds |
| `script/` | `common.sh` (environment, GPUs, deployed defense flags), `build_vectors.sh`, the `run_*.sh` drivers of section 6, `report.py` |
| `data/` / `attacks/` | datasets / fixed attack prompt resources |
| `outputs/` | generations, evaluations and defense checkpoints |
| `third_party/` | the bundled DiffuGuard generator and its provenance (`diffuguard_origin.json`) |

The LG4 and GPT-OSS result JSON layout is shared in `evaluation/results.py` and JSONL
resumption in `evaluation/streaming.py`. Atomic JSON writes for the PAP caches are in
`runtime/utils.py`.
