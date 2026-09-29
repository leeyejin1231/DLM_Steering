# DLM_Steering

## 1. 환경과 데이터 준비

명령은 저장소 루트에서 실행합니다.

```bash
uv venv --python 3.12 .venv
uv pip sync -p .venv/bin/python requirements.lock
.venv/bin/python data_downloader.py
```

AdvBench·HarmBench 다운로드에는 Hugging Face 데이터셋 접근 동의와 인증이 필요합니다.

| 벤치마크 | `--source` | 전체 행 수 |
|---|---|---:|
| JBB | `jbb_harmful` | 100 |
| HarmBench | `harmbench` | 393 |
| StrongREJECT | `strongreject` | 313 |

`--n` 기본값은 **20**입니다. 전체 평가 시 위 행 수를 명시하세요. PAP과 PAIR는 원본 데이터셋을 사용하고, DIJA만 별도 refined 프롬프트를 사용합니다.

`interface.py`의 응답 생성·PAP 준비·채점 진행률은 작업별 tqdm 하나로 표시합니다.
자식 작업의 진행 막대는 숨기고 상세 출력은 작업 로그에 보관합니다.
`exp.py`·`pap_generate.py`를 직접 실행할 때도 tqdm을 사용합니다.

### 방어 준비

| `--defense` | 설명 | 준비물 |
|---|---|---|
| `none` | 방어 없음 | 없음 |
| `ours` | 게이트·스티어링·V3 리마스킹 | 아래 체크포인트 |
| `diffuguard` | DiffuGuard | 생성기 코드가 프로젝트에 포함됨 |
| `selfreminder` | 프롬프트에 안전 안내 추가 | 없음 |

`ours`의 기본 경로에는 다음 파일이 필요합니다. 파일은 별도로 준비해야 합니다.

```text
outputs/steer_vector.pt
outputs/steer_detector.pt
outputs/gate_threshold.json
outputs/response_detector.pt
```

## 2. PAP: 공격 생성 → 방어 응답 → 채점

### 2-1. 공격 프롬프트 생성

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python pap_generate.py \
  --source jbb_harmful --seed 42 --reproduct \
  --out data/attacks/pap_better/jbb_harmful/seed42.json
```

- 기본 생성 모델: **Qwen/Qwen3-14B** (`--model`로 변경).
- 데이터마다 공격 문장을 한 번 생성합니다. 재시도와 내부 진단 채점은 없습니다.

### 2-2. 같은 공격 캐시로 두 방어 실행

기본 경로는 자동으로 선택합니다. 다른 위치의 파일을 쓰려면 `--pap-cache <파일>`로 지정할 수 있습니다.

공격 캐시는 **데이터셋·시드별로 한 번만 생성**하고, 비교할 모델과 방어가 공유합니다. 이 단계에서는 공격 생성 모델을 로드하지 않습니다.

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

`--source`, `--seed`, `--reproduct`는 캐시 생성 시 설정과 같아야 합니다. 시드 43, 44는 해당 시드의 캐시를 각각 생성해서 사용하세요. 다른 데이터셋도 동일한 순서로 실행합니다.

다른 타깃 모델에 직접 연결할 때는 JSON의 `results[*].attack_prompt`를 사용자 메시지로 전달합니다. `prompt`는 원본 평가 질문, `index`는 원본 행 식별자입니다. 캐시는 `data/` 아래에 저장된다.

## 3. 다른 공격과 실행 옵션

### DIJA

DIJA의 Qwen refined 프롬프트 세 데이터셋은 `data/dija/`에 포함되어 있습니다.
해당 자료의 출처는 [DIJA 저장소](https://github.com/ZichenWen1/DIJA)이며 라이선스는 `data/dija/LICENSE`에 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python exp.py \
  --attack dija --defense diffuguard \
  --source jbb_harmful --n 100 --seed 42 --reproduct \
  --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 \
  --out outputs/JBB-dija-diffuguard-42.json
```

DIJA의 `--gen-length` 기본값은 128입니다. 프롬프트 안의 마스크만 채우려면 `--gen-length 0`을 명시하세요.

### PAIR

PAIR는 타깃 응답을 받아 공격을 개선하므로, PAP처럼 공격 전체를 미리 생성하지 않습니다. 기본 공격 모델은 Qwen/Qwen3-14B이며 내부 성공 판정은 GCG 키워드 검사로 고정돼 있습니다. 기본 탐색 예산은 **5개 스트림 × 5회 반복**입니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python exp.py \
  --attack pair --defense diffuguard \
  --source jbb_harmful --n 100 --seed 42 --reproduct \
  --gen-length 128 --steps 128 --block-length 32 --temperature 0.2 \
  --out outputs/JBB-pair-diffuguard-42.json
```

PAIR는 타깃 공격자 모델용으로 GPU 2장을 사용합니다.

### 공통 옵션

- `--attack`: `none`, `prefix`, `dija`, `pap`, `pair`.
- `--seed`와 `--reproduct`: 난수 시드와 결정적 연산 설정. 서로 다른 GPU 아키텍처 간 비트 일치까지 보장하지는 않습니다.
- `--start`, `--n`: 원본 데이터셋에서 실행할 행 범위.
- `--gpus 0,1,2,3`: 여러 GPU로 분산 실행. PAP 타깃 생성은 GPU당 모델 1개, PAIR는 GPU를 두 장씩 묶어 사용합니다. 단일 GPU 예시의 `CUDA_VISIBLE_DEVICES=...` 대신 사용하세요.
- V3: `--remask-prompt`를 지정하면 특수 토큰을 제외한 프롬프트 전체도 리마스킹 대상에 포함합니다.
- DiffuGuard 기본값: SAR(`adaptive_step`), `repair-scope all`, threshold 0.2, refinement 8, remask ratio 0.9.

공격·방어별 상세 옵션은 선택한 조합의 도움말에서 확인합니다.

```bash
.venv/bin/python exp.py --attack pap --defense ours --help
.venv/bin/python exp.py --attack pair --defense diffuguard --help
.venv/bin/python pap_generate.py --help
```

## 4. 평가

```bash
.venv/bin/python eval_llamaguard.py \
  --in outputs/JBB-pap-v3-42.json --out outputs/JBB-pap-v3-42_lg4.json --gpus 0

.venv/bin/python run_sr_eval.py \
  --in outputs/JBB-pap-v3-42.json --out outputs/JBB-pap-v3-42_sr.json --gpu 0 --port 50001

.venv/bin/python script/report.py \
  outputs/JBB-pap-v3-42_lg4.json outputs/JBB-pap-v3-42_sr.json
```

- **LG4**: Llama-Guard-4의 unsafe 판정으로 ASR을 계산합니다.
- **GPT-OSS**: Ollama의 `gpt-oss:20b`로 StrongREJECT 채점합니다.

로컬 Ollama를 자동으로 실행해 여러 GPU에서 채점하려면 `run_sr_eval.py`에
`--auto-server --gpus 0,1 --startup-workers 2`를 지정합니다. 기본적으로 서버
2개까지 초기화하며, CUDA 탐색은 순서대로 하고 모델 로드는 겹쳐 실행합니다.
일시적인 초기화 실패는 최대 2회 재시도합니다. `--startup-workers 1`은
모델 로드까지 모두 순서대로 실행합니다. `--workers`는 로드 후 서버당 채점 동시성입니다.

### 과잉 거절·일반 능력

| 목적 | 생성 시 `--source` | 평가 명령 |
|---|---|---|
| 과잉 거절 | `truthfulqa`, `xstest_safe` 등 | `python eval_refusal.py --in <응답.json> --out <평가.json> --auto-server --gpus 0,1` (`interface.py` 평가 메뉴의 Over-refusal) |
| 일반 능력 | `mmlu`, `gsm8k`, `math500`, `truthfulqa_mc` | `python eval_utility.py --in <응답.json> --out <평가.json>` (`interface.py` 평가 메뉴의 성능) |

`gsm8k`, `math500`, `truthfulqa`는 `python data_downloader.py gsm8k math500 truthfulqa`로 `data/`에 받아 둔다. `math500`은 마지막 `\boxed{}`를 MATH 저장소의 정규화 규칙으로 정답과 비교한다.

이 평가는 `--attack none`으로 생성하고, 같은 데이터 범위에서 `--defense none`과 비교합니다. 위 명령의 Python도 `.venv/bin/python`을 사용하세요.

## 5. Dream-v0-Instruct-7B

같은 방법론(gated steering + v3 remask)을 `Dream-org/Dream-v0-Instruct-7B`에서 그대로 돌립니다.
`--model dream`을 주면 `models.py` 레지스트리에서 모델별 상수(마스크 토큰 `<|mask|>`=151666,
블록 경로 `model.model.layers`, 28층, `<|im_end|>`)가 선택되고 벡터·검출기 기본 경로가 `outputs/dream/`으로
바뀝니다. `--model`은 `common.py`가 import되기 전에 읽히므로 반드시 커맨드라인으로 줍니다(`--gpus` 샤딩 자식은
자동 전달). 환경은 LLaDA와 같은 `.venv`를 씁니다.

### 배포 설정과 체크포인트 (`outputs/dream/`)

| 산출물 | 층 | 비고 |
|---|---|---|
| `steer_detector.pt` + `gate_threshold.json` | 14 (`best_layer`) | 게이트. OOD 평균 AUROC 0.970, 임계값 2.706 |
| `steer_vector.pt` | **20** (`--layer 20` 명시) | 검증 AUROC 최고는 17이지만 alpha sweep에서 17은 alpha 1.5부터 붕괴, 20은 alpha 1에서 거절 100%·유창 |
| `response_detector3_committed_L20.pt` | **20**, 컷오프 0.387 | v3 응답 검출기. 첫 블록 경계 상태(`dlm_steering/fitting/fit_boundary_detector.py`)로 학습 |

응답 검출기 층(20)이 게이트 층(14)과 다르므로 v3의 경계 감사는 다음 forward에 편승하지 않고 별도의 무조향
forward로 검출기 층까지만 읽습니다(`dlm_steering/defenses/recovery.py`의 `_audit_piggyback`). LLaDA처럼 두
층이 같으면 이전과 동일하게 편승합니다. 체크포인트의 `layer`·`pool` 키가 이 동작을 결정하며 결과 JSON의
`defense.response_detector_layer`에 기록됩니다.

```bash
# 배포 설정 그대로 한 번 실행 (과잉 거절, XSTest-safe 250)
.venv/bin/python exp.py --model dream --attack none --defense ours --remask v3 --steer adaptive --layer 20 \
  --response-detector outputs/dream/response_detector3_committed_L20.pt \
  --remask-prompt --remask-prompt-frac 0.8 --source xstest_safe --n 250 \
  --gen-length 128 --steps 128 --block-length 32 --seed 42 --gpus 0,1 \
  --out outputs/dream/XSTest-safe-none-v3rp80-42.json
```

### 실험 스크립트

공통 설정은 `script/dream_common.sh`에 있고 환경 변수로 바꿉니다
(`TAG`=v3rp80, `SEEDS`="42 43 44", `STEER_LAYER`=20, `PROMPT_FRAC`=0.8, `RESPONSE_DETECTOR`, `GEN`/`STEPS`/`BLOCK`,
`GPUS`/`PROCS_PER_GPU`는 `common.sh`). 출력은 `outputs/dream/<SET>-<attack>-<TAG>-<seed>.json`이고 완료된 파일은 건너뜁니다.

| 스크립트 | 내용 | 후속 |
|---|---|---|
| `script/run_dream_overrefusal.sh` | XSTest-safe(250) + TruthfulQA(817) 생성 | `script/judge_dream_overrefusal.sh` (XSTest 3분류 판정 → `OR-summary-<TAG>.{json,md}`), `script/judge_dream_truthfulqa.sh` (truthful/informative → `*_tqa.json`) |
| `script/run_dream_pap.sh` | PAP 캐시 공격, temperature 0.2. 기본 JBB; `SOURCES="jbb_harmful harmbench strongreject"` | `script/eval_dream_pap.sh` (LG4 + StrongREJECT + `script/report.py`) |
| `script/run_dream_math500.sh` | MATH-500 정확도, 방어 vs `--defense none` (`DEFS`) | 내부에서 `eval_utility.py` 실행 |
| `script/run_dream_truthfulqa_mc.sh` | TruthfulQA MC1 정확도 | 내부에서 `eval_utility.py` 실행 |

응답 검출기를 다시 맞추려면 (arm별 생성 → LG4 라벨 → CPU 재적합):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dlm_steering.fitting.fit_boundary_detector --model dream --arms wildjailbreak,alpaca --gen-only
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m dlm_steering.fitting.fit_boundary_detector --model dream --arms wj_benign --gen-only
.venv/bin/python -m dlm_steering.fitting.fit_boundary_detector --model dream --pool committed --layer 20 --balanced \
  --from-samples outputs/dream/boundary_samples_{wildjailbreak,wj_benign,alpaca}.pt \
  --out outputs/dream/response_detector3_committed_L20.pt
```

벡터·게이트 검출기·임계값은 LLaDA와 같은 `MODEL=dream script/build_vectors.sh` 경로로 만듭니다. Dream에서
`--detector-layer`/`--layer`를 생략하면 각 번들의 `best_layer`를 쓰므로 벡터는 `--layer 20`을 명시합니다.

### LLaDA와 다른 점 (구현)

- Dream의 `lm_head`는 다음 위치를 예측하도록(AR shift) 학습되어 있어 `dlm_steering/runtime/models.py`의
  `ShiftedLogits` 래퍼가 `model(x).logits[0, p]`가 p번째 슬롯을 가리키도록 정렬합니다. hidden state는 위치별
  residual stream 그대로라 훅·검출기는 수정 없이 동작합니다.
- 디코딩은 Dream 자체 `diffusion_generate` 규칙의 이식(`dream_sampler.py`, `--decoder dream`이 기본값)입니다.
  블록 없이 전체 시퀀스를 `--steps` 타임스텝으로 채우고 스텝마다 `--alg`(origin 기본)로 고른 위치를 `--top-p 0.95
  --top-k 50`으로 샘플링합니다. 방어 없이 같은 seed면 네이티브 출력과 토큰 단위로 일치합니다. `--remasking`/`--schedule`은
  무시됩니다. Dream 원본 `sample_tokens`는 `Categorical(probs).sample()`을 try/except로 감싸는데, bf16 로짓에서는 확률 합
  검증(|합−1| < 1e-6)이 스텝의 약 1%에서 실패해 그 스텝만 greedy가 되고 난수도 소비하지 않습니다. `dream_sampler.py`는 이
  동작까지 재현하므로(`_categorical_would_raise`) `origin/dream` 브랜치로 만든 `outputs/dream/*-v3rp80-*` 결과와 같은 seed에서
  토큰 단위로 같습니다.
- v3의 "블록 경계"는 Dream에서는 위치가 아니라 시간 단위입니다. 답변 슬롯이 `--block-length`개 새로 확정될 때마다 그
  토큰들을 한 블록으로 감사하고, 트리거되면 그 토큰들(+`--remask-prompt` 대상)을 remask해 Dream 규칙으로
  `--recovery-steps` 타임스텝 동안 조향하며 재생성합니다(`dream_sampler.dream_denoise`).
- Dream 토크나이저는 `<|im_start|>`/`<|im_end|>`를 special로 취급하지 않아 `load_model`이 둘을 special로 등록해
  생성문에서 지워지게 합니다.
- DiffuGuard 베이스라인은 LLaDA 전용입니다(`third_party/diffuguard.py`). Dream은 저자의 별도 러너가 필요해 `exp.py`가
  `--model dream --defense diffuguard`를 거부합니다.

## 코드 구조

공격·방어·평가·실행 지원 코드는 `dlm_steering/` 아래에서 역할별로 관리합니다.
기존 CLI 명령과 `Attacker`, `Defender`, `Evaluator`, `common`, `interface` 등의
import 경로는 호환 모듈로 유지합니다. 구현을 수정할 때는 아래 패키지에서 수정하세요.

```text
dlm_steering/
├── paths.py          # 저장소·데이터 공통 경로
├── attacks/          # 공격 인터페이스, prefix, DIJA, PAP, PAIR
├── defenses/         # 방어 인터페이스, steering, V3 복구, baseline
├── evaluation/       # 평가기, JSONL 재개, 공격별 집계, 캐시, 결과 형식
├── runtime/          # 모델 로딩, 데이터, 재현성, GPU 작업 분배, JSON 저장
├── fitting/          # 벡터·검출기·임계값 학습, 거절/TruthfulQA 판정 CLI (python -m dlm_steering.fitting.<모듈>)
└── launcher/         # 대화형 입력, 실행 계획, 프로세스 실행, 진행률·결과 표시
```

| 파일·디렉터리 | 역할 |
|---|---|
| `interface.py` | 대화형 실험·평가 CLI 진입점 |
| `exp.py` / `experiment_row.py` | 실험 실행 / 행별 응답·결과 기록 |
| `eval_llamaguard.py` / `run_sr_eval.py` | LG4 / GPT-OSS 평가 CLI 진입점 |
| `pap_generate.py` / `pap_common.py` | PAP 공격 생성 / 기법 배정·캐시 검증 |
| `sampler.py` / `dream_sampler.py` | LLaDA / Dream 디퓨전 샘플러 |
| `models.py` / `model_loading.py` | 타깃 모델 설정 / 사전학습 가중치 로딩 지원 |
| `ollama_runtime.py` | 전용 Ollama 서버와 시작·종료 관리 |
| `dlm_steering/fitting/` | 벡터·검출기·임계값 학습과 거절/TruthfulQA 판정 CLI. `python -m dlm_steering.fitting.<모듈>`로 실행 |
| `dlm_steering/fitting/fit_boundary_detector.py` / `judge_truthfulqa.py` / `aggregate_dream_overrefusal.py` | Dream 첫 경계 응답 검출기 학습 / TruthfulQA truthful·informative 판정 / Dream 과잉 거절 seed 집계 |
| `script/` | 실험 조합 실행·운영 스크립트 |
| `script/dream_common.sh`, `script/run_dream_*.sh`, `script/judge_dream_*.sh`, `script/eval_dream_pap.sh` | Dream 배포 설정과 과잉 거절·PAP·MATH-500·TruthfulQA MC 실행/판정 |
| `data/` / `attacks/` | 데이터 / 고정 공격 프롬프트 자원 |
| `outputs/` | 생성·평가 결과와 방어 체크포인트 |
| `third_party/` | 프로젝트에 포함한 DiffuGuard 생성기와 출처 정보 |

LG4와 GPT-OSS의 결과 JSON 구성은 `evaluation/results.py`, JSONL 이어받기는
`evaluation/streaming.py`에서 공유합니다. 실행 계획과 PAP 캐시의 원자적 JSON
저장은 `runtime/utils.py`, 작업자 스레드 환경 설정은 `launcher/storage.py`에 모았습니다.
