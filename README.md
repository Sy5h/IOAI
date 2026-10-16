# IOAI 2026 — 풀이 코드와 한국어 해설

본선 Day 1·Day 2의 여섯 문제를 단순한 방법부터 풀어보는 학습용 구현입니다. 문제별 Python 코드, 독립 실행 노트북, 한국어 풀이와 사전 과제 연결표를 함께 정리했습니다.

**공식 정답·우승자 코드가 아닙니다.** 공식 문제문·힌트·보존된 채점 코드를 참고해 작성한 제안 풀이입니다. [2차 개선 결과](docs/improvements-round2.ko.md)에 단순한 방법으로 바꾼 코드의 A/B 점수·실행 시간·실패한 후보를 정리했습니다. 실제 대회 서버에 제출한 결과와는 구분합니다.

## 공식 공개 데이터 실측 요약

2026-10-16 측정. 현재 기본 설정의 **점수(100점 척도) / 실행 시간**이다. Order·Robot·Potato·Field는 2차 측정, Dilemma·Ghost는 변경 없이 1차 결과를 유지했다. 공개된 A/B 자료의 개발 재평가이며 실제 대회 서버 제출 결과는 아니다.

| 문제 | A 점수 / 시간 | B 점수 / 시간 |
|---|---:|---:|
| Find the Order | **77.77 / 277.7초** | **77.92 / 254.3초** |
| Robot Chasing | **66.31 / 73.2초** | **64.47 / 55.7초** |
| Potato | **47.42 / 3.7초** | **54.40 / 4.8초** |
| Double Agent Dilemma | 97.54 / 155.0초 | 97.38 / 164.7초 |
| Ghost of the Machine | 90.52 / 90.4초 | 90.81 / 85.5초 |
| IOAI Field | **91.52 / 21.3초** | **81.86 / 25.1초** |

2차에서는 Order의 탐색 폭, Robot의 주변 칸 특징, Potato의 후보 확률 혼합·직접 추측 시점, Field의 표본 안쪽 경계를 바꿨다. 큰 모델이나 새 사전학습 모델을 추가하지 않았다. 특히 Field는 A **+6.02점**, B **+9.12점**, Potato는 A **+3.17점**, B **+5.50점** 올랐다. [개선 전후 A/B 점수·시간표](docs/improvements-round2.ko.md)에서 전체 비교와 채택하지 않은 후보를 확인한다.

Field는 별도 Linux CPU runner, 나머지는 로컬 Windows PC에서 측정했다. 시간은 풀이 프로세스 시작부터 종료까지이며 다운로드·별도 채점은 제외한다. Field는 모델 학습·저장 시간이다. Order의 개선 전 시간은 이전 측정값이며 Windows 동시 작업 부하도 달라, 시간 차이를 정밀한 속도 개선으로 해석하지 않는다. [비교 조건·원자료·재현 방법](docs/improvements-round2.ko.md)을 함께 확인한다.

## 문제별 바로가기

| 문제 | 구현 | 한국어 풀이 | 제출용 노트북 | 핵심 방법 |
|---|---|---|---|---|
| Day 1-1 Find the Order | [Python](solutions/day1_find_the_order.py) | [풀이](docs/day1_find_the_order.ko.md) | [solution.ipynb](notebooks/day1_find_the_order/solution.ipynb) | prefix 고정, Whisper 전사, 화자 특징, Qwen 연결 점수, beam search |
| Day 1-2 Robot Chasing | [Python](solutions/day1_robot_chasing.py) | [풀이](docs/day1_robot_chasing.ko.md) | [solution.ipynb](notebooks/day1_robot_chasing/solution.ipynb) | 명령 파싱, 격자·상대 좌표 특징, 로봇별 MLP |
| Day 1-3 Potato | [Python](solutions/day1_potato.py) | [풀이](docs/day1_potato.ko.md) | [solution.ipynb](notebooks/day1_potato/solution.ipynb) | 부드러운 후보 갱신, 정보량 기반 질문, 후반 직접 추측 |
| Day 2-1 Double Agent Dilemma | [Python](solutions/day2_double_agent_dilemma.py) | [풀이](docs/day2_double_agent_dilemma.ko.md) | [solution.ipynb](notebooks/day2_double_agent_dilemma/solution.ipynb) | 두 모델을 동시에 고려하는 변경량 최적화, 성공 후 축소 |
| Day 2-2 Ghost of the Machine | [Python](solutions/day2_ghost_of_the_machine.py) | [풀이](docs/day2_ghost_of_the_machine.ko.md) | [solution.ipynb](notebooks/day2_ghost_of_the_machine/solution.ipynb) | 학습 경계에서 구분자 추출, BGE·로지스틱 회귀, 전환점 하나 선택 |
| Day 2-3 IOAI Field | [학습 코드](solutions/day2_ioai_field.py) · [모델](solutions/field_custom_model.py) | [풀이](docs/day2_ioai_field.ko.md) | [solution.ipynb](notebooks/day2_ioai_field/solution.ipynb) · [custom_model.py](notebooks/day2_ioai_field/custom_model.py) | 표본 안쪽 I 경계, 선형 I, 작은 O 회귀, 상수 출력, 4-bit dropout |

- [사전 과제와 본선의 연결 관계](docs/connections.ko.md)
- [2차 개선 전후 A/B 점수·시간·재현 방법](docs/improvements-round2.ko.md)
- [공식 공개 데이터 1차 실험 기록](docs/benchmark.ko.md)
- [실행·데이터·제출 방법](docs/running.ko.md)
- [검증한 범위와 남은 한계](docs/validation.ko.md)
- [공식 자료와 출처](SOURCES.md)

## 빠르게 실행하기

Python 3.10 이상에서 일반 코드와 테스트를 사용할 수 있습니다. **IOAI Field의 공식 생성기는 Linux x86-64 / CPython 3.13 전용이므로 해당 문제는 공식 런타임이 필요합니다.**

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/build_notebooks.py --check
```

실제 문제를 풀려면 공식 저장소의 해당 문제 실행 폴더에 공개 데이터와 허용 모델이 준비되어 있어야 합니다. 이 저장소에는 데이터·모델 가중치·숨겨진 정답을 넣지 않습니다. 코드가 사용하는 사전학습 모델은 로컬 경로에서만 불러옵니다.

각 문제의 `notebooks/<문제>/solution.ipynb`를 공식 문제 작업 폴더로 복사한 다음 **Run All**을 실행합니다. Field는 같은 폴더의 `custom_model.py`도 복사합니다. Potato는 대화형 문제이므로 노트북을 실행해 응답 클래스를 준비한 뒤 공식 `local_test.py`/grader로 실행합니다.

CLI 옵션은 각 Python 파일에 `--help`를 붙여 확인할 수 있습니다. 데이터 경로를 바꿀 수 있고, 시간 예산과 모델 크기·배치 크기를 조절할 수 있습니다.

## 점수에 관하여

해설에 적힌 출제자 점수는 공식 자료의 참고 기록이며, 이 저장소의 실측과 구분합니다. 실측의 원본 기록과 재현 실행기는 `benchmarks/`에 있습니다. 노트북의 시간 예산은 학습·추론을 제한하는 소프트 타이머이며, GPU 종류와 모델 로딩 속도에 따라 실제 실행 시간은 달라집니다.

정답 파일이 없는 숨겨진 평가 환경을 가정해 구현했습니다. 테스트 정답을 읽는 코드는 풀이에 포함하지 않습니다. 예외적으로 Double Agent Dilemma는 문제에서 테스트 라벨 사용을 명시적으로 허용합니다.

