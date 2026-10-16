# Day 2 — Ghost of the Machine

이 구현은 **문장마다 AI 확률을 구하고, 앞은 전부 사람·뒤는 전부 AI가 되는 경계 하나를 고르는 방법**이다. 허용된 BGE 인코더의 고정 임베딩과 로지스틱 회귀를 사용한다. 원문 문자 위치를 그대로 보존하며 학습 정답으로 경계 주변 구분자를 추정한다. 전체 공식 데이터, GPU 실행 시간, 비공개 점수는 아직 검증하지 않았다.

## 문제와 알려진 공식 점수

글마다 사람이 쓴 앞부분과 AI가 쓴 뒷부분 사이에 경계가 정확히 하나 있다. 예측값은 원문의 문자 인덱스다. 학습 데이터는 1,221개이며 공식 statement의 평가 split은 각 380개다. 양쪽은 각각 최소 180단어다.

평가식은 `100 * mean(exp(-abs(pred-true)/100))`이다. 100문자 빗나가면 해당 샘플 점수는 약 0.37이므로 원문 공백과 offset 처리가 중요하다.

공식 statement의 baseline 점수는 **28.6**, 위원회 점수는 **93.41**이다. 둘 다 leaderboard A 기준이다. 위원회는 BGE를 fine-tuning하고 문장 단위 변화점을 찾았다. **이 저장소의 고정 임베딩 구현이 93.41점을 재현했다는 뜻이 아니다.**

## 쉬운 풀이의 핵심

1. 학습 글의 정답 경계 바로 앞 문자열에서 반복되는 문장부호·공백·개행 구분자를 찾는다. 실제 문자를 확인하지 않고 마침표나 개행이라고 가정하지 않는다. 충분한 공통 구분자가 없으면 일반 문장 분할로 돌아간다.
2. 원문에서 `(start, end)` 구간을 얻는다. 원문 자체에 `.strip()`이나 공백 정규화를 적용하지 않는다. 경계를 가로지르는 학습 구간은 이진 분류 학습에서 제외한다.
3. 로컬 `bge-base-en-v1.5`의 CLS 벡터를 정규화한다. 기본 토큰 길이는 128, 배치는 64다. BGE를 업데이트하지 않고 작은 로지스틱 회귀만 학습한다.
4. 각 문장의 AI 확률 `p_i`를 얻는다. 경계가 문장 `k` 앞이라고 할 때 다음 값을 계산한다.

```text
L(k) = 앞 문장들의 log(1 - p_i) 합 + 뒤 문장들의 log(p_i) 합
```

5. `L(k)`가 큰 경계를 택한다. 학습 경계의 평균 위치를 약한 사전정보로 사용하고, 가능하면 양쪽 최소 180단어를 만족하는 후보만 비교한다.

문장부호가 없거나 가능한 범위 안에 문장 경계가 하나도 없으면, 평균 경계 비율에 가까운 **원문의 단어 시작 위치**를 선택한다. 전체 길이가 충분하면 양쪽 최소 180단어 조건을 지킨다. 글 전체가 문장 하나로 인식됐다는 이유로 0이나 글 끝을 출력하지 않는다. 시간 부족 fallback에도 같은 처리를 적용한다.

첫 번째 `p_i > 0.5` 문장을 바로 경계라고 하지 않는다. 사람 글 중 AI처럼 보이는 문장 하나가 있어도 앞뒤 문장을 함께 보고 판단하는 것이 핵심이다. 누적합만으로 계산하므로 별도의 시퀀스 모델이 필요 없다.

`--expected-score`를 켜면 후보 경계 확률에서 공식 평가식의 기대값을 최대화한다. 여러 경계의 문자 인덱스를 단순 평균내는 대신 주변 후보 확률을 합산한다. 이 옵션과 기본값의 성능 우열은 아직 검증하지 않았다.

## 데이터와 실행

필요 라이브러리는 NumPy, scikit-learn, PyTorch, Transformers다. 허용 모델은 로컬 디렉터리에 준비된 BGE-base-en-v1.5 하나다. 코드에서 인터넷으로 모델을 내려받지 않는다.

```text
dataset/train/data.jsonl
dataset/train/answers.jsonl
dataset/test_public/data.jsonl
models/bge-base-en-v1.5/...
```

Hugging Face 원본 배치처럼 정답 파일이 `train` 폴더 옆의 `train_answers.jsonl`에 있어도 읽는다. 평가 입력 파일은 `--test-dir` 아래의 `data.jsonl`이다.

```bash
python solutions/day2_ghost_of_the_machine.py --train-dir dataset/train --test-dir dataset/test_public --model-dir models/bge-base-en-v1.5 --output answers.jsonl
```

GPU 메모리가 부족하면 `--batch-size 16`부터 시도한다. 대회 환경에서는 약 16GB GPU가 제공된다. CPU에서도 선택할 수 있지만 전체 임베딩 계산이 제한 시간 안에 끝난다는 보장은 없다.

제출용 `solution.ipynb`에는 이 Python 파일 전체를 코드 셀로 넣는다. 노트북은 자체적으로 학습·추론을 수행하고 저장소 루트의 `answers.jsonl`을 만든다. 다른 Python 파일 import나 저장소에 넣어 둔 별도 학습 가중치에 의존하지 않는다. 노트북에서 실행하면 커널의 `-f` 인자를 받지 않도록 처리되어 있다.

## 검증과 실행 시간

학습 글의 15%를 **글 단위로** 검증용으로 나눈다. 같은 글의 문장들이 학습과 검증에 섞이지 않는다. 구분자 추정과 평균 경계 비율도 검증 글을 제외하고 계산한다. 출력하는 `Article-holdout score`는 이 내부 검증 점수이며 공식 leaderboard 점수가 아니다. 시간이 남으면 로지스틱 회귀를 전체 학습 문장으로 다시 학습한다.

기본 내부 시간 목표는 540초다. 임베딩 중 시간 여유가 부족하면 미리 작성한 평균 위치·구분자 기반 fallback을 남기고 종료한다. 로지스틱 회귀나 모델 로드 중에는 강제 중단하지 않으므로 이 옵션이 실제 전체 실행 시간을 보장하지는 않는다. 제출 전 실제 GPU에서 시간을 측정해야 한다.

공식 statement 제한은 학습과 추론 전체 10분, 노트북 20MB다. 보존된 원본 checker는 실행 장비 편차 보정으로 최종 B split에 1,200초를 부여하지만 이 코드의 실행 계획은 공시된 10분을 기준으로 한다.

## 출력과 알려진 한계

- 원본 checker에서는 누락·중복·알 수 없는 ID가 전체 제출 오류가 된다. 코드에서 모든 ID에 정수 경계 하나씩 있고 원문 범위 안인지 확인한다.
- 구분자가 실제 경계 패턴을 잘 못 설명하거나 문장 분할이 잘못되면 분류기만으로 복구하기 어렵다. 학습/검증의 구분자 coverage와 경계 주변 문자열을 추가 확인하는 것이 다음 개선점이다.
- 128토큰을 넘는 긴 문장은 잘린다. 배치·토큰 길이는 GPU 시간과 검증 점수를 보고 조절한다.
- frozen BGE는 단순한 출발점이다. 공식 93.41점 방법은 fine-tuning을 포함하므로 동일 성능을 기대할 근거는 없다.
- 검증 데이터 정답을 예측에 넣거나 테스트 정답을 읽지 않는다.

합성 테스트는 실제 구분자 위치 학습, 공백 앞/뒤 구분, 원문 문자 인덱스 보존, 고립된 문장 오분류를 넘는 전역 경계 선택, 글 단위 분리, 경계를 가로지르는 학습 문장 제외, 출력 ID·범위와 평가식을 확인한다. 또한 내부 문장부호가 없는 600단어 글에서 일반 추론과 시간 부족 fallback 모두 적법한 내부 단어 경계를 출력하는지 확인한다.

출처: [공식 문제와 점수](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/5_Ghost_of_Machine/statement.md), [GAITE 힌트](https://github.com/IOAI-official/IOAI-2026/blob/main/GAITE-Contest/5_Ghost_of_Machine/hint.md), [원본 checker](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/5_Ghost_of_Machine/code/grading-original/checker.py), [실제 CI 설정](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/5_Ghost_of_Machine/code/grading-original/.gitlab-ci.yml), [공식 공개 데이터](https://huggingface.co/datasets/IOAI-official/ioai-2026-ghost-of-the-machine).
