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

DIJA 공격을 위해서는 아래의 레포지토리가 필요하다 (다른 경로에 두면 `--dija-dir`로 지정).

```bash
git clone https://github.com/ZichenWen1/DIJA.git   # 레포 루트에 DIJA/ 생성
```

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
| `--attack` | `none`, `prefix`, `dija`(논문 refined 프롬프트), `dija_template`(구 합성 템플릿), `pap`(미구현), `pair`(미구현) |
| `--defense` | `none`, `ours`, `proposed`, `selfreminder`(미구현), `diffuguard`(미구현) |
| `--source` | 유해: `jbb_harmful`, `advbench`, `harmbench`, `strongreject`, `xstest_unsafe` / 무해(over-refusal): `truthfulqa`, `xstest_safe`, `jbb_benign`, `wj_benign` / 일반화(accuracy): `mmlu`, `gsm8k`, `truthfulqa_mc` |
| 공통 | `--n`, `--start`, `--steps`, `--gen-length`, `--block-length`, `--temperature`, `--remasking`, `--schedule {const,linear,cosine}`, `--seed`, `--reproduct`, `--gpus` |

`--gen-length` 기본값은 128, dija/dija_template는 0.

### DIJA 공격 (논문 재현)

`--attack dija`는 `DIJA/run_<bench>/refine_prompt/*_refined_Qwen.json`의 refined 프롬프트를 `--source`의 vanilla
프롬프트로 찾아 그대로 쓴다 (`jbb_harmful`, `harmbench`, `strongreject`만 지원, 세 세트 모두 100% 매칭).
원본 `*_llada.py`와 같게 `<mask:N>`을 마스크 N개로 펼쳐 user turn 안에 넣고, 어시스턴트 턴은 붙이지 않으며
(gen_length 0), 마스크 하나를 한 스텝에 채우고(`--dija-steps auto`, 프롬프트마다 steps = 마스크 수), temperature 0.2를 쓴다.
채점 텍스트는 원본처럼 vanilla 프롬프트와 겹치는 토큰 prefix 뒤부터 assistant 헤더 앞까지, 즉 채워진 템플릿이다.
`--temperature`, `--gen-length`, `--dija-steps N`을 주면 원본에서 벗어난 설정으로 돌릴 수 있다.

```bash
python exp.py --attack dija --defense none --source jbb_harmful --n 100 --out outputs/JBB-dija-none-42.json
python exp.py --attack dija --defense none --source harmbench --n 393 --out outputs/HarmBench-dija-none-42.json
python exp.py --attack dija --defense none --source strongreject --n 313 --out outputs/SR-dija-none-42.json
python exp.py --attack dija --defense ours --remask v3 --source jbb_harmful --n 100 --out outputs/JBB-dija-v3-42.json
```

전용 인자 목록: `python exp.py --attack dija --defense ours --help`

### 재현 모드

기본값은 off, 결과를 빠르게 보고 싶을 때는 그대로 쓰고, 동일한 결과를 재현해야 할 때 켠다:
속도는 대략 2배 정도 차이난다.
--seed로 값을 고정할 수 있으며, 기본값은 42이다.
```bash
python exp.py --source jbb_harmful --n 20 --reproduct --out outputs/base.json
```

### 멀티 GPU (`--gpus`)

`--gpus 0,1,...`를 주면 프롬프트를 GPU 수만큼 나눠 GPU당 자식 프로세스를 띄운다.
(`CUDA_VISIBLE_DEVICES` 자동 지정)

### `--defense ours` 주요 인자

```bash
python exp.py --defense ours \
    --detector outputs/steer_detector.pt --detector-layer 18 \
    --vector outputs/steer_vector.pt --layer 25 --alpha 1.0 \
    --steer adaptive --remask v2
```

- `--steer {none,fixed,adaptive,triggered}` — `none`: 스티어링 없음, `fixed`: non-adaptive, `adaptive`: 매 스텝 연속 게이트
- `--remask {none,v2,v3}` — `v2`: 기본값, llada_steering_remasking_v2 방식, `none`: 리마스킹 없음
  - `--steer triggered`(v3 전용): v3 응답 검출기가 트리거되기 전에는 steering을 걸지 않고, 트리거 뒤 복구와 남은 블록에서만 adaptive 게이트로 건다. 무해 프롬프트에서 게이트가 거의 항상 열려 생기는 over-refusal을 피하려는 옵션
  - `v3`: 블록 경계마다 로지스틱 회귀 응답 검출기(`--response-detector`, 기본 `outputs/response_detector.pt`)가 커밋된 토큰을 채점하고, 첫 경계에서 트리거되면 해당 블록 + 프롬프트 안의 채워진 스팬(DIJA)을 전부 remask한 뒤 `--recovery-steps`(기본 32)만큼 재생성
- `--alpha`, `--transform {additive,project}`, `--gate-threshold`, `--gate-width`, `--remask-trigger` 등은 사용성 개편할 계획.
  (`--initial-only`는 제거됨 — `--steer triggered`와 같이 쓰면 step-0에서 감시가 꺼져 스티어링이 영구히 잠기는 조합 버그가 있었다.)


### Over-refusal

무해 세트에 방어를 걸어 생성한 뒤 `judge_refusal.py`로 거절률을 잰다. 무해 세트는 target이 없으므로 `--attack none`으로 돌린다.

```bash
python exp.py --attack none --defense none --source truthfulqa --n 200 --out outputs/TQA-none-none-42.json
python exp.py --attack none --defense ours --remask v3 --source truthfulqa --n 200 --out outputs/TQA-none-v3-42.json
python steering/judge_refusal.py --in outputs/TQA-none-v3-42.json --out outputs/TQA-none-v3-42_judged.json

# XSTest: safe 250개는 over-refusal, unsafe 200개는 대조용 정상 거절률
python exp.py --attack none --defense ours --remask v3 --source xstest_safe --n 250 --out outputs/XSTest-safe-none-v3-42.json
python exp.py --attack none --defense ours --remask v3 --source xstest_unsafe --n 200 --out outputs/XSTest-unsafe-none-v3-42.json
python steering/judge_refusal.py --in outputs/XSTest-safe-none-v3-42.json --out outputs/XSTest-safe-none-v3-42_judged.json
```

### 일반화 성능 (utility)

방어를 걸었을 때 일반 능력이 얼마나 깎이는지 정확도로 잰다. 프롬프트에 정답 키(`answer`, `task`)가 같이 실려
결과 JSON에 남고, `eval_utility.py`가 생성문에서 답을 뽑아 채점한다. 방어 없는 baseline(`--defense none`)과 같은
`--n`으로 돌려 비교한다.

| source | n | 형식 | 정답 |
|---|---|---|---|
| `mmlu` | 14042 (57과목 test, seed 0 고정 셔플이라 `--n`이 과목 혼합 샘플) | 4지선다, 문자로 답 | A~D |
| `gsm8k` | 1319 (test) | 단계별 풀이 후 `#### <숫자>` | 숫자 |
| `truthfulqa_mc` | 817 | MC1 방식: 최선 답 + 오답들을 항목별 셔플 | 문자 |

gsm8k는 풀이가 길어 `--gen-length 256`을 권장한다. 답을 못 뽑은 생성(거절 등)은 오답으로 세고 `unparsed`로 따로 센다.

```bash
python exp.py --attack none --defense none --source mmlu --n 500 --out outputs/MMLU-none-none-42.json
python exp.py --attack none --defense ours --remask v3 --source mmlu --n 500 --out outputs/MMLU-none-v3-42.json
python eval_utility.py --in outputs/MMLU-none-v3-42.json --out outputs/MMLU-none-v3-42_acc.json

python exp.py --attack none --defense ours --remask v3 --source gsm8k --n 300 --gen-length 256 --out outputs/GSM8K-none-v3-42.json
python eval_utility.py --in outputs/GSM8K-none-v3-42.json --out outputs/GSM8K-none-v3-42_acc.json

python exp.py --attack none --defense ours --remask v3 --source truthfulqa_mc --n 817 --out outputs/TQAmc-none-v3-42.json
python eval_utility.py --in outputs/TQAmc-none-v3-42.json --out outputs/TQAmc-none-v3-42_acc.json
```


## 평가

| 스크립트 | 평가자 | 용도 |
|---|---|---|
| `eval_llamaguard.py` | Llama-Guard-4-12B (GPU) | safe/unsafe 판정 → ASR |
| `run_sr_eval.py` | gpt-oss:20b (ollama) | StrongREJECT 루브릭 → ASR |
| `steering/judge_refusal.py` | gpt-oss:20b (ollama) | XSTest 3-way → over-refusal |
| `eval_utility.py` | 규칙 기반 (GPU 불필요) | mmlu/gsm8k/truthfulqa_mc 정답 추출 → accuracy |

```bash
python eval_llamaguard.py --in outputs/dija_ours.json --out outputs/dija_ours_lg4.json --gpus 0,1,2,3
python run_sr_eval.py --in outputs/dija_ours.json --out outputs/dija_ours_sr.json --gpus 4,5,6,7
python steering/judge_refusal.py --in outputs/base.json --out outputs/base_judged.json --gpus 4,5
```

멀티 GPU: 세 스크립트 모두 `--gpus`로 아이템을 나눠 병렬 채점한다 (`eval_utility.py`는 CPU라 해당 없음).

- `eval_llamaguard.py`는 GPU당 자식 프로세스가 각자 Llama-Guard-4를 로드.
- `run_sr_eval.py`/`judge_refusal.py`는 샤드마다 ollama 컨테이너를 따로 띄운다 — `--port`부터 i씩 증가한 포트,
  컨테이너명 `ollama-<port>`, 각각 `--gpus`의 i번째 GPU에 바인딩 (podman 필요. 실행 후 컨테이너는 남는다).
- 셋 다 `--start/--n`으로 부분 구간만 돌릴 수 있고, 파트별 `.jsonl`로 스트리밍/재개된다.


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