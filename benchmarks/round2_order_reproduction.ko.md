# Order 2차 실험 재현

변경은 기본 beam 크기 **16→256 한 줄**이다. Whisper-small·Qwen2.5-0.5B, 전사 설정, 화자 조건, LM batch 4, 540초 소프트 예산은 그대로다. 과거 기준선과 이번 새 실행은 서로 다른 시점의 측정이므로 작은 시간 차이를 이 변경의 속도 효과로 해석하지 않는다.

## 데이터와 기준 소스

저장소 최상위에서 실행한다. Python 의존성과 `huggingface_hub`가 필요하다. 공개 데이터·정답·모델은 Git에서 제외한 `work/`에만 둔다.

```powershell
python benchmarks/download_order_robot.py order-data --workspace work/bench-order-robot
python benchmarks/download_order_robot.py order-eval --workspace work/bench-order-robot
python benchmarks/download_order_robot.py models --workspace work/bench-order-robot
New-Item -ItemType Directory -Force work/round2 | Out-Null
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/Sy5h/IOAI/991a49b48092286967b5bd7cada465e336ebc4af/solutions/day1_find_the_order.py', 'work/round2/order_baseline.py')"
$benchPath = (Resolve-Path work/bench-order-robot).Path
```

위 명령은 셸의 줄바꿈·BOM 변환 없이 원본 바이트를 저장한다. 원래 기준 소스 SHA256은 `1474a40641722150965baf3d2d531166f28a34d9ebf150c618be7d3a51baecbe`이다. 현재 코드에 `--beam 16`을 명시해도 같은 기준 알고리즘을 재현한다.

## 공개 pretrain에서 선택 과정 재현

다음 cache 명령은 GPU에서 모델을 실행할 수 있다. 개발용 20대화만 읽고 전사·Qwen 점수를 계산한다. search는 저장된 점수로 CPU에서 8개 설정을 비교하며, 이 단계만 공개 pretrain 정답을 읽는다. 캐시는 제출물이나 최종 풀이 입력이 아니다.

```powershell
python benchmarks/round2_order_develop.py cache --source work/round2/order_baseline.py --data-dir work/bench-order-robot/order/development/dataset/test_public --models-dir work/bench-order-robot/models --output work/round2/order-dev-cache.json
python benchmarks/round2_order_develop.py search --source work/round2/order_baseline.py --cache work/round2/order-dev-cache.json --truth work/bench-order-robot/scorer-only/order/pretrain_answers.json --output work/round2/order-search.json --selection-output work/round2/order-selection.json
```

개발 캐시는 LM batch 8과 사용 가능한 모든 연결을 사용했다. 최종 실행의 LM batch 4와 수치가 조금 달라질 수 있다. 따라서 캐시 기준선 75.8797은 과거 전체 실행의 75.5781과 별도 결과다. 캐시 search의 0.01~0.12초에는 모델 전사·추론이 포함되지 않는다. `first10`, `last10`은 선택에 사용한 같은 20개 표본의 두 부분이며 독립 holdout이 아니다.

기록된 8개 후보의 평균에서 beam 256이 80.0778로 가장 높았다. 설정은 최종 평가 전에 `order-selection.json`으로 고정했다. 이 파일의 실제 선택 근거는 결과 JSON의 `selection`에 보존한다. 이미 공개된 A/B에 맞춰 개발 후보를 다시 고르는 재현은 하지 않는다.

## 새 프로세스의 전체 실행

재현 실행기의 작업 디렉터리가 바뀌므로 모델 경로는 위에서 구한 절대 경로를 사용한다. 새 run 디렉터리를 사용해야 한다. A 실행 예시는 다음과 같다.

```powershell
python benchmarks/reproduce_order_robot.py run --task order --label round2_beam256_a --script solutions/day1_find_the_order.py --dataset work/bench-order-robot/order/test_leaderboard_a/dataset --truth work/bench-order-robot/scorer-only/order/test_leaderboard_a_answers.json --run-dir work/round2/order-runs/leaderboard_a --data-revision 71a35b5a566665e242416eba45f018dc378663b8 --split test_leaderboard_a --device cuda --threads 4 --timeout 630 -- --budget 540 --beam 256 --whisper "$benchPath/models/whisper-small" --qwen "$benchPath/models/qwen2.5-0.5b"
```

B는 `leaderboard_a`를 모두 `leaderboard_b`로 바꾼다. pretest는 `--dataset work/bench-order-robot/order/official`, `--truth work/bench-order-robot/scorer-only/order/pretest_answers.json`, `--split public_pretest`, 새 `--run-dir`를 사용한다. `order/pretest/dataset`은 이 다운로더가 만드는 경로가 아니다.

과거 기준선의 새 실행도 필요하면 `--beam 16`을 주고 다른 run 디렉터리를 쓴다. 게시된 baseline 시간은 새로 돌린 paired 측정이 아니라 과거 1차 실측이다. 각 final 실행은 캐시를 쓰지 않고 전사부터 답 저장까지 다시 수행한다. `--truth`는 별도 부모 scorer에만 전달되고, 풀이 인자나 환경변수에는 전달되지 않는다.

## 공식 채점식 교차 확인과 게시 기록

고정 revision `e1963801aa4cf33a5a03520b39fa6620a7894c54`의 [원본 check.py](https://github.com/IOAI-official/IOAI-2026/blob/e1963801aa4cf33a5a03520b39fa6620a7894c54/Individual-Contest/1_Find_the_Order/code/grading-original/check.py)를 `work/scorer-audit/order.py`에 준비한다. collector가 SHA256을 확인하고 `validate_permutation`, `score_one`, `score_submission` 함수만 실행한다. Podman·GPU 자원 제한·노트북 변환까지 수행하는 공식 전체 grader 실행은 아니다.

```powershell
python benchmarks/round2_order_collect.py --base-dir work/bench-order-robot --round-dir work/round2 --baseline-source work/round2/order_baseline.py --official-scorer work/scorer-audit/order.py --workspace-root . --output work/round2/reproduced_order_results.json
```

collector는 원래 비교 폴더인 `runs/order_baseline_pretest`, `runs/order_baseline_a`, `runs/order_baseline_b`와 이번 `order-runs/pretest-corrected`, `order-runs/leaderboard_a`, `order-runs/leaderboard_b`를 사용한다. 자신의 새 실행 폴더를 수집하려면 해당 매핑을 맞춰야 한다. 과거 로그를 새 측정인 것처럼 바꾸지 않는다. 원본 정답·추론 답 배열·오디오·전사문·모델 가중치·캐시 내용은 게시하지 않는다.

게시 JSON에는 초기 pretest 경로 오류도 실패로 남긴다. 입력 읽기 전에 끝난 그 실행에는 점수가 없으며, 경로를 고친 실행만 성공 점수로 비교한다. traceback의 작업 경로와 Python 설치 경로도 익명화한다.
