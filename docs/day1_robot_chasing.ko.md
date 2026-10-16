# Day 1 · Robot Chasing

공식 공개 데이터를 이용해 실제 실행한 풀이입니다. 언어를 간단한 규칙으로 파싱하고, 관찰 가능한 정보를 숫자로 바꾼 뒤 로봇별 작은 분류기 6개를 학습합니다. 학습 데이터 안의 독립 holdout에서는 histogram gradient boosting(HGB)이 개선됐지만 최종 A/B 결과가 혼재하여 기본값은 기존 MLP를 유지합니다. HGB 후보는 `--model hist`로 사용할 수 있습니다. 아래 점수는 원대회 컨테이너가 아닌 로컬 Windows 환경에서 측정한 값입니다.

## 문제와 핵심 관찰

6개의 로봇이 각각 일관된 행동 패턴을 가집니다. 입력은 벽을 포함한 `8×8×2` 격자, 로봇 ID, 방향, 지시문, 들고 있는 물체입니다. 다음 행동 하나를 예측해야 합니다. 각 행은 독립적인 스냅샷이므로 이전 행을 직전 상태로 취급하면 안 됩니다.

학습은 60,000개 행, 로봇당 10,000개입니다. 평가 입력은 3,600개 행이며 로봇별 정확도를 평균합니다. 문제문 기준 5분에 학습과 추론을 모두 끝내야 합니다. 행동은 `0 위, 1 아래, 2 왼쪽, 3 오른쪽, 4 집기, 5 놓기`입니다. [공식 문제문](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/2_Robot_Chasing/statement.md)

가장 유용한 요령은 자연어를 거대한 언어 모델에 맡기지 않고, 동사·색·물체를 추출하는 것입니다. 모든 단어와 문장 패턴은 학습 데이터에도 등장하므로 파서를 점검하기 쉽습니다. 또한 정답은 로봇이 실제로 선택한 행동입니다. 최단경로 계산만으로 로봇의 선호까지 설명할 수는 없습니다.

## 구현한 방법

1. `train/labels.json`만 읽어 로봇별 최빈 행동을 계산하고 평가 행 순서대로 즉시 저장합니다.
2. 지시문의 색-물체 쌍과 동사로 접근·집기·옆에 놓기를 구분합니다. 두 물체가 나오면 첫 번째를 출발 물체, 두 번째를 목적 물체로 유지합니다.
3. 전체 격자의 물체·색 one-hot과 구조적 특징을 계산합니다. 목표뿐 아니라 다른 물체와 토큰도 특징 계산에 사용합니다. 빈칸·벽의 색은 의미가 없으므로 색 특징을 0으로 둡니다. MLP는 전체 특징을 사용하고 HGB 후보는 아래의 구조적·단어 특징 176개를 사용합니다.
4. 로봇 좌표, 목표 상대 좌표·거리, 방향, 들고 있는 물체, 앞칸 물체, 네 이웃칸, 물체-색별 개수·가장 가까운 위치를 추가합니다. `put-next`에서 출발 물체를 들고 있는지와 앞에 있는지도 표시합니다.
5. 파서가 놓친 표현에 대비해 작은 고정 해시 기반 단어 빈도 특징도 넣습니다.
6. HGB 후보는 로봇별 `HistGradientBoostingClassifier`입니다. 최대 128회, leaf 31개, learning rate 0.1, L2 1을 사용합니다. 학습 자료의 15%를 내부 early stopping 검증용으로 사용합니다. 모델 종류를 고르는 데 쓴 별도 holdout과는 구분됩니다.
7. 로봇별로 학습·추론한 결과를 저장합니다. 예산이 부족하면 아직 학습하지 않은 로봇은 최빈 행동을 유지합니다. 기본 MLP는 `특징 → 192 → 64 → 행동 6개` 구조이며, 로봇별 epoch를 번갈아 학습하고 내부 검증에서 최빈 행동보다 좋은 가중치만 채택합니다.

파서·좌표 특징·소지품·로봇별 분류기는 [공식 GAITE 힌트](https://github.com/IOAI-official/IOAI-2026/blob/main/GAITE-Contest/2_Robot_Chasing/hint.md)의 방향입니다. 하이퍼파라미터와 상세 특징은 이 구현의 선택이며 최적값이라는 뜻은 아닙니다.

## 실행

```text
dataset/train/observations.json
dataset/train/labels.json
dataset/test_public/observations.json
```

평가 폴더의 `labels.json`은 읽지 않으며 존재할 필요도 없습니다.

```bash
# 표준 라이브러리만 필요한 최빈 행동 baseline
python solutions/day1_robot_chasing.py --epochs 0

# 기본 MLP, 270초 소프트 예산으로 학습·추론
python solutions/day1_robot_chasing.py --budget 270

# holdout에서 선택한 HGB 후보와 비교
python solutions/day1_robot_chasing.py --model hist

# tree 모델의 계산량 조절
python solutions/day1_robot_chasing.py --model hist --hist-iterations 64 --cpu-threads 4
```

HGB 학습에는 `numpy`, `scikit-learn`과 그 의존성 `threadpoolctl`이 필요합니다. MLP에는 `numpy`, `torch`가 필요하고 CUDA가 없으면 CPU를 사용합니다. 인터넷, 외부 모델, 추가 학습 데이터는 사용하지 않습니다. 공식 제출용 독립 notebook에 코드를 포함한다면 `main([])`을 호출해 Jupyter 실행 인자가 argparse에 들어가지 않게 하십시오.

## 놓치기 쉬운 조건

이동 행동은 절대 방향으로 먼저 몸을 돌리고 그다음 이동을 시도합니다. 벽이나 물체에 막혀도 방향은 바뀝니다. 따라서 **막힌 방향으로 이동하는 행동을 무조건 제거하면 오답이 될 수 있습니다.** 코드도 행동을 이런 이유로 마스킹하지 않습니다. 집기·놓기는 현재 방향의 바로 앞칸에 적용됩니다.

물체를 들고 있으면 원래 위치에서 해당 물체가 사라질 수 있습니다. 코드는 목표 좌표가 없어도 고정 길이 특징을 만들고, 소지품 일치 여부를 별도로 전달합니다. 동일한 색·종류 물체가 여러 개면 HGB 후보의 위치 요약은 가장 가까운 것을 사용하며 개수를 함께 전달합니다. 다른 중복 물체들의 정확한 좌표가 필요한 경우 정보가 줄어드는 한계가 있습니다. 기본 MLP에는 전체 격자도 전달합니다.

로봇마다 절대 방향 또는 색·물체 선호가 있을 수 있습니다. 검증 없이 격자 회전이나 색 교환을 데이터 증강으로 적용하지 않았습니다. 파서는 흔한 영어 표현을 처리하는 휴리스틱이므로 학습 지시문 중 색-물체 쌍이 추출되지 않는 사례를 확인하고 필요하면 확장해야 합니다.

## 점수와 한계

모델 선택에는 공식 train 60,000행만 사용했습니다. 로봇별로 20%를 seed `20261016`으로 따로 떼어 48,000행 학습 / 12,000행 holdout을 만들었습니다. 이 holdout 정답은 실행 프로세스에 전달하지 않고 예측 후 scorer에서만 읽었습니다.

| 학습 내 독립 holdout 비교 | 정확도(로봇별 평균, %) | 프로세스 벽시계 |
|---|---:|---:|
| 기존 MLP | 64.7333 | 143.10초 |
| ExtraTrees, 304개 특징, 128 trees | 65.1500 | 109.80초 |
| **holdout에서 선택한 HGB, 176개 특징, 최대 128회** | **66.5083** | **87.22초** |

각 설정을 한 번씩 실행했습니다. CPU 4 threads, Intel Core Ultra 9 185H, Python 3.11, scikit-learn 1.7.2 환경이며 다운로드·scorer 시간은 제외했습니다. 다른 작업이 동시에 실행될 수 있는 로컬 환경이므로 시간 차이는 정밀한 전용 하드웨어 성능 비교로 해석하지 마십시오. 모델 선택을 마친 뒤 전체 train 60,000행으로 다시 학습하고 공식 A/B를 별도로 평가했습니다. A/B 정답으로 하이퍼파라미터를 조정하지 않았습니다.

| 최종 공개 평가 | 기존 MLP 정확도 / 시간 | holdout 선택 HGB 정확도 / 시간 |
|---|---:|---:|
| Leaderboard A, 3,600행 | **64.9167%** / 98.00초 | 63.7500% / 100.77초 |
| Leaderboard B, 3,600행 | 63.9444% / 97.53초 | **64.5278%** / 96.10초 |

유지한 기본 MLP의 `public/pretest` 600행 결과는 **63.0000% / 100.92초**입니다. 전체 train 60,000행으로 학습하고, 이 축약 평가에서도 별도 새 프로세스를 사용했습니다. 이 실행은 예외 처리 보완 후의 최종 소스로 완료했습니다.

**최종 평가에서 일관된 개선은 확인되지 않았습니다.** HGB는 holdout에서 +1.7750%p였지만 A에서는 -1.1667%p, B에서는 +0.5833%p였습니다. HGB를 무조건 더 강한 정답으로 설명하면 안 됩니다. 독립 평가 결과가 혼재하므로 최종 배포의 기본값은 기존 MLP를 유지했습니다. 두 방법 모두 옵션으로 제공하며, 새로운 환경에서는 학습 자료 안에서 검증해야 합니다. 이 결과를 본 뒤 추가 튜닝이나 앙상블 선택을 하지 않았습니다. 기본값 유지 결정은 A/B 확인 후 이루어졌으므로 그 결정까지 사전에 고정됐다고 주장하지 않습니다.

데이터는 Hugging Face의 revision `8a82acbdfda387269649d4bcf580fbdab1ed3cc6`에 고정했습니다. `public/pretest` 600행은 원래 공개 시험의 일부이고, 위 표는 그것보다 큰 `private/test_leaderboard_a`, `private/test_leaderboard_b`입니다. 현재 공개된 대회 후 평가 자료를 사용했습니다. 원대회 Linux 컨테이너를 실행한 결과는 아닙니다.

기준 소스는 commit `b3e60ae8e14b55d97c44214dc1ec7d8e49295a89`입니다. [실측 JSON](../benchmarks/results_order_robot.json), [재현 실행기](../benchmarks/reproduce_order_robot.py), [고정 revision 다운로드 도구](../benchmarks/download_order_robot.py), [탐색 후보 재현 코드](../benchmarks/robot_candidates.py)에 명령·버전·소스 해시·로그를 기록했습니다. 정상 완료 측정 뒤에는, 후속 로봇 학습에서 예외가 발생해도 이미 저장한 예측을 보존하도록 예외 처리만 보완했습니다. 이 경로는 위 정상 완료 실행에서는 사용되지 않았습니다.

[공식 채점 함수 교차 검증](../benchmarks/scoring_audit_order_robot.json)에서는 고정 revision의 원본 `score_submission`을 저장된 예측에 적용했습니다. 이 문제 8회와 Find the Order 11회, 총 19회 모두 점수가 소수점 4자리까지 일치했습니다. Podman 실행·GPU 할당·notebook 변환·시간 제한을 포함한 공식 전체 grader를 실행한 검증은 아닙니다.

공개된 공식 grader runbook에는 다음 **출제 측 참고 점수**가 기록되어 있습니다.

| 출제 측 방법 | Leaderboard A | Leaderboard B |
|---|---:|---:|
| 항상 위로 이동 | 20.81 | 20.58 |
| 출제자 풀이: 로봇별 MLP | 72.03 | 71.14 |

이는 이번 코드의 결과가 아닙니다. 그 출제자 모델의 상세 구현이 공개된 것은 아니므로 같은 점수를 재현했다고 주장하지 않습니다. [공식 기록](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/2_Robot_Chasing/code/grading-original/RUNBOOK.md)

예산에는 JSON 읽기와 특징 계산도 포함합니다. 기본 270초에서 마지막 20초를 추론에 남기지만, 시간 제한은 배치 사이에서 확인하는 소프트 제한입니다. 시작 단계의 파일 읽기나 한 번의 GPU 연산을 강제 중단하지 않습니다. 모든 하드웨어에서 대회 제한 300초를 통과한다는 보장은 없습니다. 데이터 전처리 중 예산이 소진되면 최빈 행동 답을 반환합니다.

학습 중 출력되는 `validation`은 학습 자료 안의 내부 검증 점수이며 위의 외부 holdout/A/B와 구분해야 합니다. 이 코드에는 검증되지 않은 앙상블을 추가하지 않았습니다.

## 일반 clone에서 재현

앞서 설명한 Python 의존성과 다운로드용 `huggingface_hub`를 준비한 다음 저장소 최상위에서 실행합니다. 데이터 다운로드 revision은 코드에 고정되어 있습니다.

```bash
python benchmarks/download_order_robot.py robot --workspace work/bench-order-robot
python benchmarks/reproduce_order_robot.py run --task robot --label mlp_a --script solutions/day1_robot_chasing.py --dataset work/bench-order-robot/robot/test_leaderboard_a/dataset --truth work/bench-order-robot/scorer-only/robot/test_leaderboard_a_answers.json --run-dir work/bench-order-robot/runs/mlp_a --data-revision 8a82acbdfda387269649d4bcf580fbdab1ed3cc6 --split test_leaderboard_a --device cpu --threads 4 --timeout 330 -- --model mlp --budget 270 --cpu-threads 4
```

B는 `test_leaderboard_a`와 `mlp_a`를 각각 `test_leaderboard_b`, `mlp_b`로 바꿉니다. HGB 후보는 `--model hist`와 새로운 실행 폴더 이름을 사용합니다. 평가 정답은 `scorer-only`에 분리되고 `--truth`는 예측 완료 후 채점하는 실행기에서만 읽습니다. 풀이 프로세스에는 학습 자료와 평가 관측만 들어갑니다. 재실행에는 예측 파일이 없는 새 `--run-dir`을 지정하십시오. 실제 실행 기록 JSON은 측정 당시 명령을 보존하며, 위 명령은 일반 clone에 맞춘 재현 예시입니다.
