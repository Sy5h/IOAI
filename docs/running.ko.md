# 실행·데이터·제출 방법

## 노트북으로 실행

1. [공식 문제 폴더](https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest)의 README를 열고 공개 데이터·허용 모델·실행 환경을 준비합니다.
2. 이 저장소의 `notebooks/<문제>/solution.ipynb`를 해당 공식 작업 폴더로 복사합니다.
3. IOAI Field는 같은 디렉터리의 `custom_model.py`도 복사합니다.
4. 노트북의 마지막 실행 셀 인자를 확인하고 Run All을 실행합니다. Potato의 실행 방식은 아래 예외를 따릅니다.
5. 공식 공개 평가기로 점수·출력 형식·실행 시간을 확인합니다. 실제 대회에서는 노트북이 숨겨진 데이터에서 다시 실행됩니다.

이 저장소의 코드가 사용하는 기본 경로는 공식 대회 원본 환경을 기준으로 합니다. 별도 Colab 교육 버전은 경로가 다를 수 있으므로 실행 인자로 지정하세요. 데이터셋 전체나 모델 가중치를 GitHub에 올리지 않습니다.

| 문제 | 읽는 기본 경로 | 생성 결과 | 문제문 실행 제한 |
|---|---|---|---|
| Find the Order | `dataset/test_public`, `models/whisper-small`, `models/qwen2.5-0.5b` | `answers.json` | 600초 |
| Robot Chasing | `dataset/train`, `dataset/test_public` | `predictions.json` | 300초 |
| Potato | `dataset/vocabulary.json`, `dataset/public_embeddings.npy` | stdin/stdout 응답 | 전체 120게임 600초 |
| Double Agent Dilemma | `dataset/test_public`, `models` | `submission.zip` | 720초 |
| Ghost of the Machine | `dataset/train`, `dataset/test_public`, `models/bge-base-en-v1.5` | `answers.jsonl` | 600초 |
| IOAI Field | `data/train_config`, 공식 `core` 생성기 | `model.pt` | 300초 |

실제 스크립트는 제한보다 짧은 소프트 예산을 기본값으로 둡니다. 모델 로딩이나 한 번의 GPU 연산 자체를 중간에 강제 종료하지는 않습니다.

## Python 스크립트로 실행

공개 A/B 전체를 실제로 실행한 점수·시간과 채택하지 않은 개선 실험은 [공식 데이터 실측](benchmark.ko.md)에 정리했습니다. `benchmarks/`의 실행기는 데이터 다운로드·풀이 실행·채점을 분리하며 제출용 노트북에는 포함되지 않습니다. Field는 저장소 Actions의 수동 workflow로 Linux 환경에서 재현할 수도 있습니다.

아래 명령은 공식 데이터 작업 폴더에서 실행한다고 가정합니다. `/path/to/IOAI`는 저장소 경로로 바꾸세요. Windows에서도 Python 인자는 같으며 절대 경로 형식만 바꾸면 됩니다.

```bash
python /path/to/IOAI/solutions/day1_find_the_order.py --help
python /path/to/IOAI/solutions/day1_robot_chasing.py --help
python /path/to/IOAI/solutions/day1_potato.py --help
python /path/to/IOAI/solutions/day2_double_agent_dilemma.py --help
python /path/to/IOAI/solutions/day2_ghost_of_the_machine.py --help
python /path/to/IOAI/solutions/day2_ioai_field.py --help
```

문제별 상세 실행 명령은 각 한글 풀이를 참고하세요. Find the Order의 `--prefix-only`와 Robot Chasing의 `--epochs 0`은 학습 없이 입력·출력 파이프라인을 먼저 점검하는 옵션입니다. 고득점 옵션은 아닙니다.

## Potato의 대화형 실행

Potato는 실행 후 stdin으로 심판 메시지를 받아야 합니다. 터미널에서 스크립트만 실행하면 메시지를 기다리는 것이 정상입니다. 공식 `local_test.py`가 있는 폴더에 노트북을 놓고 다음과 같이 실행합니다.

```bash
python local_test.py solution.ipynb --limit 5
```

공개 테스트의 심판은 공개 임베딩을 사용하므로 실제 비공개 표현과의 차이까지 검증하지 못합니다. 노트북에서 Run All을 해도 stdin을 기다리지 않도록 구성했습니다. 공식 grader가 Python으로 변환해 실행할 때만 대화 루프를 시작합니다.

## IOAI Field의 환경과 저장

공식 생성기는 보호된 런타임에 의존합니다. 일반 Windows Python 3.11에서 전체 문제를 실행할 수 있다고 가정하지 마세요. 공식 Linux x86-64 / CPython 3.13 환경에서 생성기를 준비한 뒤 실행해야 합니다.

제출 모델 클래스의 모듈 이름은 `custom_model`이어야 합니다. 생성된 노트북 옆의 `custom_model.py`를 그대로 사용하세요. `solutions/field_custom_model.py`는 개발용 원본이며, 이 이름으로 pickle된 모델을 그대로 제출하는 것은 피해야 합니다.

## 소스를 수정한 뒤 노트북 갱신

```bash
python tools/build_notebooks.py
python tools/build_notebooks.py --check
python -m unittest discover -s tests -v
```

노트북은 소스 코드를 자체 포함합니다. Python 파일만 고치면 기존 노트북 내용은 바뀌지 않으므로 반드시 다시 생성하세요. 생성기는 자동 실행 guard를 제거하고 문제별 실행 셀을 한 번만 넣습니다. Field의 보조 모델 파일도 함께 갱신합니다.

## 제출 전 확인

- 각 문제의 출력 파일 이름과 테스트 행/ID 수가 일치하는지 확인합니다.
- Find the Order는 재생 순서가 아니라 파일별 rank 배열을 제출합니다.
- Robot은 관측 행 순서를 그대로 지킵니다.
- Ghost는 전처리 문자열이 아니라 원문 문자 위치를 제출합니다.
- 공격 ZIP의 텐서는 원본 이미지 크기이며 ZIP 최상위에 놓입니다.
- Field는 전체 파라미터 수가 20,260개 미만이어야 합니다.
- 공개 검증 정답을 읽는 평가 코드를 제출용 풀이 셀에 추가하지 않습니다.

