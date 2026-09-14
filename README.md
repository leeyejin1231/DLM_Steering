# DLM_Steering

## 0. 환경 세팅

```bash
uv venv --python 3.12 .venv              # uv 없으면: python3.12 -m venv .venv
uv pip sync -p .venv/bin/python requirements.lock   # 재현용
```

## 1. 데이터 준비

```bash
python data_downloader.py   # data/에 jbb_harmful.csv, advbench.parquet, harmbench.parquet 저장
```

아래 페이지에서 접근 동의 후 huggingface-cli login을 세팅해야 다운로드된다:

- https://huggingface.co/datasets/walledai/AdvBench
- https://huggingface.co/datasets/walledai/HarmBench

! 추가로, ./outputs/ 폴더에 steer_detector.pt, steer_vector.pt가 있어야하고, 만약 v3로 defense를 하려는 경우에는 response_detector.pt도 있어야 한다.

## 실험 실행

모든 실험은 `exp.py`로 실행 - `--attack`과 `--defense`를 고르면 선택한 공격 및 방어에 맞는 인자값을 넣어야 함.

### 벤치마크만 평가
```bash
# JBB
python exp.py --attack none --defense none --source jbb_harmful --n 100 --out outputs/JBB-none-none-42.json
# AdvBench
python exp.py --attack none --defense none --source advbench --n 520 --out outputs/AdvBench-none-none-42.json
# HarmBench
python exp.py --attack none --defense none --source harmbench --n 393 --out outputs/HarmBench-none-none-42.json
```

### 공격만 평가
```bash
# Dija
python exp.py --attack dija --defense none --source jbb_harmful --n 100 --out outputs/JBB-dija-none-42.json
# Prefix
python exp.py --attack prefix --defense none --source jbb_harmful --n 100 --out outputs/JBB-prefix-none-42.json
# PAP
python exp.py --attack pap --defense none --source jbb_harmful --n 100 --out outputs/JBB-pap-none-42.json
# PAIR
python exp.py --attack pair --defense none --source jbb_harmful --n 100 --out outputs/JBB-pair-none-42.json
```

### 공격+방어 평가
```bash
# llada_steering_remasking_v2.py와 같은 방어.
python exp.py --attack dija --defense ours --remask v2 --source jbb_harmful --n 100 --out outputs/JBB-dija-v2-42.json
# llada_steering_v2.py와 같은 방어.
python exp.py --attack dija --defense ours --steer fixed --remask none --source jbb_harmful --n 100 --out outputs/JBB-dija-steer-42.json
# run_proposed_dija.py와 같은 방어.
python exp.py --attack dija --defense proposed --source jbb_harmful --n 100 --out outputs/JBB-dija-proposed-42.json
# remasking 전용 detector를 추가한 버전
python exp.py --attack dija --defense ours --remask v3 --source jbb_harmful --n 100 --out outputs/JBB-dija-v3-42.json
```

| 옵션 | 값 |
|---|---|
| `--attack` | `none`, `prefix`, `dija`, `pap`(미구현), `pair`(미구현) |
| `--defense` | `none`, `ours`, `proposed`, `selfreminder`(미구현), `diffuguard`(미구현) |
| `--source` | `jbb_harmful`, `advbench`, `harmbench` |
| 공통 | `--n`, `--start`, `--steps`, `--gen-length`, `--block-length`, `--temperature`, `--remasking`, `--schedule {const,linear,cosine}`, `--reproduct` |

`--gen-length` 기본값은 128 dija는 0

전용 인자 목록: `python exp.py --attack dija --defense ours --help`

### 재현 모드

기본값은 off, 결과를 빠르게 보고 싶을 때는 그대로 쓰고, 동일한 결과를 재현해야 할 때 켠다:
속도는 대략 2배 정도 차이난다.
--seed로 값을 고정할 수 있으며, 기본값은 42이다.
```bash
python exp.py --source jbb_harmful --n 20 --reproduct --out outputs/base.json
```

### `--defense ours` 주요 인자

```bash
python exp.py --defense ours \
    --detector outputs/steer_detector.pt --detector-layer 18 \
    --vector outputs/steer_vector.pt --layer 25 --alpha 1.0 \
    --steer adaptive --remask v2
```

- `--steer {none,fixed,adaptive}` — `none`: 스티어링 없음, `fixed`: non-adaptive, `adaptive`: 매 스텝 연속 게이트
- `--remask {none,v2,v3}` — `v2`: 기본값, llada_steering_remasking_v2 방식, `none`: 리마스킹 없음
  - `v3`: 블록 경계마다 로지스틱 회귀 응답 검출기(`--response-detector`, 기본 `outputs/response_detector.pt`)가 커밋된 토큰을 채점하고, 첫 경계에서 트리거되면 해당 블록 + 프롬프트 안의 채워진 스팬(DIJA)을 전부 remask한 뒤 `--recovery-steps`(기본 32)만큼 재생성
- `--alpha`, `--transform {additive,project}`, `--gate-threshold`, `--gate-width`, `--remask-trigger`, `--initial-only` 등은 사용성 개편할 계획.

## 평가

| 스크립트 | 평가자 | 용도 |
|---|---|---|
| `eval_llamaguard.py` | Llama-Guard-4-12B (GPU) | safe/unsafe 판정 → ASR |
| `run_sr_eval.py` | gpt-oss:20b (ollama) | StrongREJECT 루브릭 → ASR |
| `steering/judge_refusal.py` | gpt-oss:20b (ollama) | XSTest 3-way → over-refusal |

```bash
python eval_llamaguard.py --in outputs/dija_ours.json --out outputs/dija_ours_lg4.json
python run_sr_eval.py --in outputs/dija_ours.json --out outputs/dija_ours_sr.json --port 50001 --gpu 0
python steering/judge_refusal.py --in outputs/base.json --out outputs/base_judged.json
```

## 벡터/디텍터 (`steering/`)

```bash
python steering/fit_vector.py          # steering 벡터 추출 → outputs/steer_vector.pt
python steering/fit_detector.py        # 디텍터 벡터 학습 → outputs/steer_detector.pt
python steering/pick_threshold.py      # 게이트 threshold 선택 → outputs/gate_threshold.json
python steering/check_detector.py      # 디텍터 AUROC 확인
```

## 이전 버전대비 명령어 전환
- `llada.py` -> `python exp.py --attack none --defense none`
- `llada_steering_v2.py` -> `python exp.py --defense ours --steer fixed --remask none`
- `llada_steering_remasking_v2.py` -> `python exp.py --defense ours --remask v2`
- `run_proposed_dija.py` -> `python exp.py --attack dija --defense proposed`
- remasking 전용 detector를 추가한 버전은 `--remask v3`