# Potato 2차 실험 재현

저장소 루트에서 실행한다. 데이터·정답·모델·원본 로그는 Git에서 제외한 `work/`에 보관한다. NumPy만 필요하며 GPU를 사용하지 않는다. 공개 데이터 준비는 기존 `download_potato_ghost.py --only potato`를 사용한다.

먼저 고정 기준 소스를 준비한다. 아래는 Bash 예시다. PowerShell에서는 같은 Git 객체를 UTF-8 파일로 저장한다.

```bash
mkdir -p work/round2/potato
git show 991a49b48092286967b5bd7cada465e336ebc4af:solutions/day1_potato.py > work/round2/potato/baseline.py
python benchmarks/download_potato_ghost.py --data-root work/bench-potato-ghost --only potato
```

개발 실행기는 공개 임베딩·단어 목록·공개 target 제외 목록만 읽는다. `private/` 폴더를 받는 인자 자체가 없다. 12개 후보를 먼저 비교하고, 선택한 `mixture_guess_8`과 기준선만 겹치지 않는 128단어·다른 잡음 seed에서 확인한다. 첫 단계 결과가 아래 선택과 다르면 그 결과도 그대로 남기고, 이미 알려진 A/B 수치에 맞춰 후보를 고르면 안 된다.

```bash
python benchmarks/round2_potato_develop.py --data-dir work/bench-potato-ghost/potato-source/public --baseline work/round2/potato/baseline.py --output work/round2/potato/development_64.json --count 64
python benchmarks/round2_potato_develop.py --data-dir work/bench-potato-ghost/potato-source/public --baseline work/round2/potato/baseline.py --output work/round2/potato/confirmation_128.json --count 128 --seed 20261018 --exclude-development work/round2/potato/development_64.json --candidates baseline mixture_guess_8
```

최종 공개 A/B 재평가는 별도 실행기에서 진행한다. `--evaluation-dir`는 심판에만 전달된다. 자식 풀이 프로세스는 공개 임베딩과 정상 비교 응답만 받는다. 이는 OS 보안 격리가 아닌 연구용 프로세스 분리다.

```bash
python benchmarks/round2_potato_benchmark.py --baseline work/round2/potato/baseline.py --candidate solutions/day1_potato.py --data-dir work/bench-potato-ghost/potato-source/public --evaluation-dir work/bench-potato-ghost/evaluation-assets/potato/private --output-dir work/round2/potato/evaluation
python -m unittest discover -s tests -p test_round2_potato.py
python -m unittest discover -s tests -p test_potato_ghost.py
```

`baseline/candidate × A/B` 네 실행이 각각 별도 solver 프로세스를 쓴다. 출력 JSON의 `run_seconds`는 solver 생성 직전부터 종료까지이며 import·공개 행렬 준비·120게임을 포함한다. 심판의 비공개 행렬 읽기·정답 읽기·다운로드는 이 시간 밖이다. 분할별 1회 측정이며 CPU 4 threads를 고정한다. 동시 작업과 OS 캐시 때문에 소수점 이하 시간은 달라질 수 있다.

실행기는 A/B 실행 전에 후보 소스 해시와 개발 근거를 `selection_frozen.json`에 기록한다. 기존 기록이 있으면 덮어쓰지 않는다. 재실험은 새 폴더를 쓰고 과거 선택 파일을 새 선택 기록인 것처럼 복제하지 않는다. 게시 결과를 만드는 collector는 per-game records와 target hash를 제외하고 경로를 익명화한다. 기존 게시 측정값을 보존하려면 다른 출력 파일을 사용한다.

```bash
python benchmarks/round2_potato_collect.py --work-dir work/round2/potato --workspace-root . --output work/round2/potato/reproduced_results.json
```

합성 개발 점수는 공식 점수가 아니다. 현재 공개 A/B는 이전에도 관찰했던 자료이므로 이번 고정안의 결과 역시 개발 재평가로 설명한다. 후보 선택 이후 A/B 추가 탐색은 하지 않았다.
