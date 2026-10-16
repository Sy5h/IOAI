# Vision 문제 재현

모든 명령은 이 저장소 최상위에서 실행한다. 데이터·가중치·생성한 ZIP은 `work/`에만 보관하고 Git에 추가하지 않는다. 점수 원자료는 `results_vision.json`에 기록한다.

## Double Agent Dilemma

측정 환경은 Python 3.11.0, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, timm 1.0.22, RTX 4070 Laptop 8 GB, PyTorch CPU threads 4였다. 이미 CUDA 지원 torch/torchvision이 설치된 환경에서 다음과 같이 실행한다.

```bash
python -m pip install timm==1.0.22
python benchmarks/prepare_dilemma.py --work-dir work/bench-vision
python benchmarks/prepare_dilemma.py --work-dir work/bench-vision --verify-only
python benchmarks/benchmark_dilemma.py --assets work/bench-vision/dilemma --source solutions/day2_double_agent_dilemma.py --split test_leaderboard_a --seconds 660 --learning-rate 0.0001 --output work/bench-vision/a-improved
python benchmarks/benchmark_dilemma.py --assets work/bench-vision/dilemma --source solutions/day2_double_agent_dilemma.py --split test_leaderboard_b --seconds 660 --learning-rate 0.0001 --output work/bench-vision/b-improved
```

원본 비교에는 [기존 저장소 commit의 소스](https://github.com/Sy5h/IOAI/blob/b3e60ae8e14b55d97c44214dc1ec7d8e49295a89/solutions/day2_double_agent_dilemma.py)를 `work/bench-vision/baseline_dilemma.py`로 저장하고 `--source`를 그 파일로, `--learning-rate`를 `0.001`로 바꾼다. 고정된 공개 train 첫 20장을 확인하려면 `--split train --limit 20`을 사용한다. 공식 공개 test는 `--split test_public`이다.

다운로더는 `dilemma_assets.json`의 고정 URL·크기·SHA-256을 검증한다. `private/test_leaderboard_a`, `private/test_leaderboard_b`라는 경로도 해당 revision에서 이미 공개된 파일이다. 미공개 서버에 접근하거나 실제 대회 리더보드에 제출하는 작업은 없다.

원본 공식 `evaluate.py`를 수정 없이 불러온다. 학습률 선택에 쓰인 train 첫 20장, fallback 개발의 참고가 된 public 100장, 이후 공개 A/B를 구분한다. 예약 없는 첫 fallback은 A에서 점수가 올랐지만 B에서 한 공격이 실패하며 원본보다 낮아졌다. 그 뒤 첫 시도 65%·재시도 예약 35%의 예산 보완을 한 번 적용했으므로 최종 A/B는 개발 재평가다. A/B 정답을 각 이미지 공격 목표로 쓰는 것은 문제에서 명시적으로 허용한다.

최종 명령의 기본값은 `--first-attempt-fraction 0.65`다. `--first-attempt-fraction 1`로 첫 fallback 정책을 재현할 수 있다. 최초의 작은 학습률 단독 실험은 원본 commit 소스에 `--learning-rate 0.0001`을 적용한 것이다. 원본·작은 학습률 단독·예약 없는 fallback·시간 예약안을 JSON에서 구분하여 보존한다.

최종 A/B는 각각 97.5372/97.3802점, 154.954/164.672초이며 모든 공격 400개가 성공했다. 원본보다 점수는 각각 0.3476/0.3321점 올랐고 시간은 2.19/2.38배로 늘었다. 최종 예약 코드를 `test_public`에 다시 실행한 결과는 없다.

`run_seconds`는 cold Python process의 imports·첫 CUDA 초기화·모델 로딩·공격·ZIP 생성을 포함한다. `scoring_seconds`는 별도 공식 채점 시간이다. 다운로드·패키지 설치는 제외한다. 최초 train 원본 측정만 사전 clean accuracy 확인으로 GPU가 준비된 상태였으며 다른 cold 측정의 시간과 직접 비교하지 않는다.

## IOAI Field

공식 보호 runtime은 Linux x86-64 CPython 3.13에서 실행한다. Windows에서 대체 생성기를 만들어 점수를 추정하지 않는다. 저장소 Actions의 **Public IOAI Field CPU benchmark → Run workflow**로 표준 무료 CPU runner에서 재현할 수 있다. workflow는 수동 실행 전용이며 `contents: read`, 30분 제한이고 secrets를 사용하지 않는다.

지원 환경에서 직접 실행하는 명령은 다음과 같다.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.3 matplotlib==3.10.6
python tools/benchmark_field.py --work-dir work/field-benchmark
```

공식 generator·scorer·baseline과 데이터 revision을 고정한다. 처음에는 공개 `train_config`의 별도 seed 1729·8071에서 학습 상한 2,500·10,000·25,000을 비교했으며 2,500이 선택됐다. 그 실험은 JSON의 `step_count_experiment`에 보존한다.

현재 하네스는 2,500 steps를 유지하고 `(영역당 문턱 보정 표본, I safety) = (1024,0.5), (16384,1.5), (65536,2.5)`를 비교한다. 후보 선택에만 public seed 1729·8071, 영역당 8192점의 생성 검증셋을 쓴다. 동점이면 보정 표본 수가 작은 것을 선택한다. 그 뒤 선택값을 고정하고 public의 원래 평가 설정 및 공개된 A/B 설정에서 매번 처음부터 재학습한다. 각 설정에 대해 공식 baseline, 기존 저장소 소스, `reliability_candidate`를 따로 평가한다. 이 보완은 앞선 B에서 배경 점에 큰 I 값을 출력한 오류를 본 뒤 만든 것으로, A/B 재측정은 독립 확증이 아닌 개발 검증이다.

선택된 최종 기본값은 `--router-steps 2500 --o-steps 2500 --validation-per-region 16384 --i-safety 1.5`다. 큰 public 생성 검증 평균은 70.5762에서 78.2887로 올랐고, 최종 공개 A는 89.5119→85.4997, B는 68.1464→72.7411이었다. A 하락과 B 상승을 함께 보고한다. 이전 문턱은 `--validation-per-region 1024 --i-safety 0.5`로 재현할 수 있다.

HF 경로명이 `private/leaderboard_*_config`여도 이 revision에 공개된 JSON만 내려받는다. README의 B 복수 설정 설명과 달리 실제 공개 A/B 파일은 각각 단일 `field_config.json`이다. 개수는 실행 결과에도 기록한다. JSON 안의 정답 생성 규칙은 학습 입력으로 허용된 설정이며, 별도 평가 표본은 공식 scorer만 생성한다.

`training.wall_seconds`는 Python subprocess 전체 학습 시간, `scoring.wall_seconds`는 별도 채점 subprocess 시간이다. 패키지 설치·자료 준비와 분리한다. 각 모델은 265초 자체 학습 예산을 사용하고, 큰 단일 연산까지 강제 중단하는 제한은 아니다. 매 실행의 명령·소스 SHA-256·실제 step 수·지역별 점수·파라미터 수·장비 정보는 로그와 JSON에 남긴다.
