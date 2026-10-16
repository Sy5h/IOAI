# Day 1 · Robot Chasing

공식 공개 데이터를 이용해 실제 실행한 풀이입니다. 언어를 간단한 규칙으로 파싱하고 로봇별 작은 MLP 6개를 학습합니다. 2차 개발에서는 원본 전체 격자를 유지하면서 로봇 주변 5×5 정보와 집기·놓기 조건을 추가했습니다. 은닉층 크기와 학습 설정은 그대로입니다. 기본값은 `--features local`이며 `--features base`로 기존 MLP 특징을 사용할 수 있습니다. 기존 HGB 후보도 `--model hist`로 남겨 두었습니다. 아래 점수는 원대회 컨테이너가 아닌 로컬 Windows 환경에서 측정한 값입니다.

## 문제와 핵심 관찰

6개의 로봇이 각각 일관된 행동 패턴을 가집니다. 입력은 벽을 포함한 `8×8×2` 격자, 로봇 ID, 방향, 지시문, 들고 있는 물체입니다. 다음 행동 하나를 예측해야 합니다. 각 행은 독립적인 스냅샷이므로 이전 행을 직전 상태로 취급하면 안 됩니다.

학습은 60,000개 행, 로봇당 10,000개입니다. 평가 입력은 3,600개 행이며 로봇별 정확도를 평균합니다. 문제문 기준 5분에 학습과 추론을 모두 끝내야 합니다. 행동은 `0 위, 1 아래, 2 왼쪽, 3 오른쪽, 4 집기, 5 놓기`입니다. [공식 문제문](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/2_Robot_Chasing/statement.md)

가장 유용한 요령은 자연어를 거대한 언어 모델에 맡기지 않고, 동사·색·물체를 추출하는 것입니다. 모든 단어와 문장 패턴은 학습 데이터에도 등장하므로 파서를 점검하기 쉽습니다. 또한 정답은 로봇이 실제로 선택한 행동입니다. 최단경로 계산만으로 로봇의 선호까지 설명할 수는 없습니다.

## 구현한 방법

1. `train/labels.json`만 읽어 로봇별 최빈 행동을 계산하고 평가 행 순서대로 즉시 저장합니다.
2. 지시문의 색-물체 쌍과 동사로 접근·집기·옆에 놓기를 구분합니다. 두 물체가 나오면 첫 번째를 출발 물체, 두 번째를 목적 물체로 유지합니다.
3. 전체 격자의 물체·색 one-hot과 구조적 특징을 계산합니다. 목표뿐 아니라 다른 물체와 토큰도 특징 계산에 사용합니다. 빈칸·벽의 색은 의미가 없으므로 색 특징을 0으로 둡니다. 기본 MLP는 원래 1,328개 특징에 주변 격자 250개와 상호작용 조건 44개를 더한 1,622개를 사용합니다. HGB 후보는 원래의 구조적·단어 특징 176개를 유지합니다.
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

# 기본 MLP: 전체 격자 + 주변 5x5, 270초 소프트 예산
python solutions/day1_robot_chasing.py --budget 270

# 기존 MLP 특징을 그대로 사용
python solutions/day1_robot_chasing.py --features base

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

## 1차 실측: 기존 특징과 HGB

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

**1차 최종 평가에서 HGB의 일관된 개선은 확인되지 않았습니다.** HGB는 holdout에서 +1.7750%p였지만 A에서는 -1.1667%p, B에서는 +0.5833%p였습니다. HGB를 무조건 더 강한 정답으로 설명하면 안 됩니다. 결과가 혼재하여 1차 배포에서는 기존 MLP를 유지했습니다. 당시 A/B 확인 후 추가 튜닝이나 앙상블 선택을 하지 않았습니다. 기본값 유지 결정은 A/B 확인 후 이루어졌으므로 그 결정까지 사전에 고정됐다고 주장하지 않습니다. 다음 절은 새 요청으로 진행한 2차 개발 기록입니다.

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

## 2차 개발: 전체 격자에 작은 주변 창 추가

기준 commit은 `991a49b48092286967b5bd7cada465e336ebc4af`입니다. 기존 48,000행 학습 / 12,000행 holdout을 그대로 사용했습니다. 지시문 파서는 학습 표현을 처리했지만, 기존 MLP가 주변 장애물·목표·방향을 결합해서 집기·놓기 직전 상태를 알아내야 하는 부담이 있었습니다. 이 조건을 직접 표시하는 간단한 특징을 비교했습니다.

| 학습 내부 holdout, 각 1회 | 정확도 | 새 프로세스 벽시계 |
|---|---:|---:|
| 기준 MLP, 1,328개 특징 | 64.7333% | 92.509초 |
| 경로·상호작용 특징 HGB, 404개 | 66.5083% | 164.996초 |
| 경로·상호작용 특징만 MLP, 404개 | 60.9750% | 85.768초 |
| 13×13 상대 격자로 대체한 MLP, 1,866개 | 58.8167% | 79.643초 |
| **원본 + 주변 5×5 MLP, 1,622개: 선택** | **66.0833%** | **88.522초** |
| 원본 + 절대 목표 마스크 MLP, 1,518개 | 64.1417% | 92.010초 |

선택안은 기준 MLP보다 +1.3500%p였습니다. HGB 경로 후보는 0.4250%p 더 높았지만 실행이 오래 걸렸고 1차 HGB의 holdout 점수와 같았습니다. 작은 MLP를 유지하는 후보 중 점수와 실행 시간을 고려해 주변 5×5 추가를 선택했습니다. 원본을 대체한 후보들의 하락은 이 표현 방식과 MLP의 조합이 덜 효과적이었다는 결과이며, 필요한 정보가 반드시 사라졌다는 인과 증명은 아닙니다.

추가한 5×5 창은 로봇을 중심에 두되 위·아래·왼쪽·오른쪽의 절대 축을 유지합니다. 채널은 출발 물체, 목적 물체, 벽, 일반 물체, 토큰 6색으로 10개입니다. 원본 전체 격자도 함께 들어가므로 먼 물체를 버리지 않습니다. 추가 44개 수치는 네 방향의 앞칸이 비었는지, 출발 물체를 집을 수 있는지, 놓으면 목적 물체 옆이 되는지, 현재 그쪽을 보는지, 운반 단계와 상대 좌표를 표시합니다. 행동을 강제로 정하거나 막힌 이동을 제거하지 않습니다. BFS는 탐색 후보에만 사용했으며 최종 기본 풀이에는 넣지 않았습니다.

은닉층 `192 → 64`, batch 512, learning rate 0.002, 최대 60 epochs, seed 2026을 유지했습니다. 선택안은 A/B 실행 전에 고정했습니다. A/B는 이미 1차에서 사용한 공개 자료이므로 이번 결과는 **새로운 숨겨진 시험이 아니라 개발 재평가**입니다. 평가 정답은 예측 후 scorer에서만 읽고, A/B로 학습하거나 설정을 다시 조정하지 않았습니다.

| 고정 후 개발 재평가, 각 3,600행 | 기준 MLP 정확도 / 시간 | 선택한 주변 창 MLP 정확도 / 시간 | 점수 변화 |
|---|---:|---:|---:|
| Leaderboard A | 64.9167% / 111.938초 | **66.3056% / 73.192초** | **+1.3889%p** |
| Leaderboard B | 63.9444% / 66.031초 | **64.4722% / 55.707초** | **+0.5278%p** |

원본과 선택안 모두 각 평가를 새 프로세스로 실행했습니다. 네 번 모두 로봇 6개 전부를 학습했고 fallback은 0개였습니다. 양쪽 점수가 올라 선택한 `local` 기본값을 유지했습니다. 모든 로봇의 개별 정확도가 오른 것은 아니며 표의 수치는 공식 로봇별 평균 정확도입니다. A/B를 본 뒤 특징·학습 설정을 추가 조정하지 않았습니다.

모든 측정은 CPU 4 threads이며 GPU를 사용하지 않았습니다. 동시 실행된 Order 추론과 다른 작업의 부하가 CPU 시간에 영향을 줄 수 있습니다. 단일 실행 벽시계에는 import·입력·특징·학습·추론을 포함하고 다운로드·채점은 제외했습니다. 시간 차이를 정밀한 속도 향상 배수로 해석하면 안 됩니다.

[2차 전체 실측·공식 점수 대조](../benchmarks/round2_robot_results.json), [새 프로세스 재현기](../benchmarks/round2_robot_benchmark.py), [후보 비교 코드](../benchmarks/round2_robot_candidate.py), [경로 특징 후보](../benchmarks/round2_robot_features.py)에 성공·탈락 후보를 모두 보존했습니다. 총 10회 모두 고정 revision의 공식 채점 함수와 소수점 4자리까지 일치했습니다. 이는 채점 수식 대조이며 원대회 전체 실행 환경을 재현한 것은 아닙니다. 실제 학습 관측 200개에서 선택 후보와 독립 제출 소스의 특징이 비트 단위로 같음을 확인했고, 새 합성 테스트 9개와 기존 테스트 17개가 통과했습니다. 합성 테스트 결과는 대회 점수가 아닙니다.

## 일반 clone에서 재현

앞서 설명한 Python 의존성과 다운로드용 `huggingface_hub`를 준비한 다음 저장소 최상위에서 실행합니다. 데이터 다운로드 revision은 코드에 고정되어 있습니다.

```bash
python benchmarks/download_order_robot.py robot --workspace work/bench-order-robot
python benchmarks/round2_robot_benchmark.py --label selected_local_mlp_a --script solutions/day1_robot_chasing.py --dataset work/bench-order-robot/robot/test_leaderboard_a/dataset --truth work/bench-order-robot/scorer-only/robot/test_leaderboard_a_answers.json --run-dir work/round2/robot/runs/selected_local_a --split test_leaderboard_a -- --model mlp --features local --budget 270 --cpu-threads 4
```

B는 명령의 `_a`를 `_b`로 바꿉니다. 기존 MLP 특징과 비교하려면 `--features base`와 새 실행 폴더 이름을 사용합니다. HGB는 `--model hist`로 실행하며 `--features`와 관계없이 기존 176개 특징을 사용합니다. 평가 정답은 `scorer-only`에 분리되고 `--truth`는 예측 완료 후 채점하는 실행기에서만 읽습니다. 풀이 프로세스에는 학습 자료와 평가 관측만 들어갑니다. 재실행에는 예측 파일이 없는 새 `--run-dir`을 지정하십시오. 실제 실행 기록 JSON은 측정 당시 명령을 보존하며, 위 명령은 일반 clone에 맞춘 재현 예시입니다.

측정 당시 기준 commit의 파일 자체를 비교하려면 Git clone에서 아래처럼 복원합니다. 이후 위 실행 명령의 `--script`를 `work/round2/robot/baseline.py`로 바꾸고 `--features local`을 빼며, 새로운 label/run-dir을 사용합니다.

```bash
python -c "from pathlib import Path; import subprocess; p=Path('work/round2/robot/baseline.py'); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(subprocess.check_output(['git','show','991a49b48092286967b5bd7cada465e336ebc4af:solutions/day1_robot_chasing.py']))"
```

학습 내부 후보 선택을 재현하려면 같은 분할을 만들고 후보 실행기를 사용합니다. `--variant`를 `paths-hist`, `paths-compact-mlp`, `relative-grid-mlp`, `local-grid-mlp`, `target-grid-mlp`로 바꾸되 각각 새 실행 폴더를 지정하십시오. 아래 후보 실행기는 현재 소스에도 호환되며 중복 특징 추가를 방지합니다.

```bash
python benchmarks/reproduce_order_robot.py prepare-holdout --source work/bench-order-robot/robot/official/train --destination work/bench-order-robot/robot/holdout/dataset --truth-dir work/bench-order-robot/scorer-only/robot --seed 20261016 --fraction 0.2
python benchmarks/round2_robot_benchmark.py --label local_grid_mlp_holdout --script benchmarks/round2_robot_candidate.py --dataset work/bench-order-robot/robot/holdout/dataset --truth work/bench-order-robot/scorer-only/robot/holdout_answers.json --run-dir work/round2/robot/runs/local_grid_mlp_holdout --split train_holdout_seed20261016 -- --variant local-grid-mlp --budget 270 --cpu-threads 4
```
