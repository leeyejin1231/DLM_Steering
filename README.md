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
| `steering/` | 벡터·검출기 학습과 거절 평가. `python -m steering.<모듈>`로 실행 |
| `script/` | 실험 조합 실행·운영 스크립트 |
| `data/` / `attacks/` | 데이터 / 고정 공격 프롬프트 자원 |
| `outputs/` | 생성·평가 결과와 방어 체크포인트 |
| `third_party/` | 프로젝트에 포함한 DiffuGuard 생성기와 출처 정보 |

LG4와 GPT-OSS의 결과 JSON 구성은 `evaluation/results.py`, JSONL 이어받기는
`evaluation/streaming.py`에서 공유합니다. 실행 계획과 PAP 캐시의 원자적 JSON
저장은 `runtime/utils.py`, 작업자 스레드 환경 설정은 `launcher/storage.py`에 모았습니다.
