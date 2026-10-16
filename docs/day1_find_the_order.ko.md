# Day 1 · Find the Order

제안 풀이입니다. 공식 정답 코드나 입상자의 검증된 최고 성능 풀이가 아닙니다. 아래 프로그램은 출력 형식과 핵심 계산을 합성 데이터로 검사했으며, 실제 대회 음성·GPU 전체 평가 점수는 측정하지 않았습니다.

## 문제와 채점

영어 대화가 7–20개의 음성 파일로 나뉘어 섞여 있습니다. `prefix.json`이 정확한 첫 두 파일 번호를 알려 줍니다. 출력은 재생 순서 목록이 아니라 `P[파일 번호] = 원래 발화 위치`인 순열입니다. 예를 들어 재생 순서 `[2, 0, 1]`의 출력은 `[1, 2, 0]`입니다.

모든 두 발화의 선후관계가 맞는 비율을 계산한 뒤 대화별로 평균합니다. 문제문 기준 실행 제한은 10분이며 학습도 포함됩니다. 제공 환경은 GPU 약 16GB, 인터넷 없음, 제출 `solution.ipynb` 1MB 이하입니다. 허용 사전학습 모델은 wav2vec2, Whisper, Qwen2.5-0.5B입니다. 코드의 기본 경로는 실제 원본 baseline에 있는 Whisper-small과 Qwen입니다. [공식 문제문](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/1_Find_the_Order/statement.md)

## 가장 간단한 부분점수

첫 두 발화를 고정하고 나머지만 섞으면 길이 `n`인 대화의 기대 점수는 다음과 같습니다.

```text
1 - (n-2)*(n-3) / (2*n*(n-1))
```

`n=10`이면 약 68.89%입니다. 무작위 섞임을 가정한 수학적 기대값이며 개별 대화의 보장 점수가 아닙니다. 프로그램의 prefix baseline은 나머지를 파일 번호순으로 둡니다. 입력 파일 번호 자체가 무작위이므로 같은 기대값을 갖지만, 한 번의 실제 평가에서는 달라집니다. 이 방식은 모델 없이도 즉시 유효한 답을 만듭니다. 공식 원본 baseline은 첫 두 발화도 활용하지 않고 그대로 출력합니다. [원본 baseline](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/1_Find_the_Order/code/baseline-original/solution.ipynb)

## 구현한 방법

1. 시작하자마자 모든 대화의 prefix baseline을 `answers.json`에 저장합니다.
2. Whisper-small로 각 발화를 영어로 전사합니다. GPU에서는 FP16, 배치 8, greedy decoding을 기본으로 사용합니다.
3. MFCC 평균·표준편차로 두 화자를 나눕니다. prefix를 두 그룹의 기준으로 삼습니다. 그룹 크기가 교대 발화 수와 일치하고 모든 점이 충분히 분리될 때만 화자 교대를 탐색 조건에 넣습니다.
4. Qwen으로 발화 `j`가 `i` 다음에 등장하는 평균 토큰 로그확률을 계산합니다. `j`가 단독으로 나오는 평균 로그확률을 빼서, 짧고 흔한 문장이 무조건 유리해지는 영향을 줄입니다. 실제로 발화 토큰에 해당하는 부분만 점수에 포함하고 문맥·padding은 제외합니다.
5. 첫 두 발화를 고정한 beam search로 연결 점수 합이 큰 순서를 찾습니다. 기본 beam 크기는 16입니다.
6. 순서를 rank 배열로 뒤집어 저장합니다. 완성된 대화마다 출력 파일을 갱신합니다.

고정 prefix와 확실한 화자 교대로 도달할 수 없는 연결은 Qwen으로 계산하지 않습니다. 단독 발화 점수를 빼는 보정은 beam의 중간 후보 선택에 영향을 줍니다. 완성된 순서에서는 모든 남은 발화가 한 번씩 등장하므로 그 보정 항의 합이 동일합니다. 따라서 이 보정이 완성된 순서의 목적함수 자체를 바꾸는 것은 아닙니다.

음성 전사, MFCC 화자 구분, prefix 활용, 화자 교대 탐색은 [공식 GAITE 힌트](https://github.com/IOAI-official/IOAI-2026/blob/main/GAITE-Contest/1_Find_the_Order/hint.md)의 방향입니다. **로그확률 정규화·단독 확률 보정·beam 크기는 이 구현의 제안**이며 공식 힌트가 지정한 공식이 아닙니다.

## 실행

필요한 파일은 다음과 같습니다. 평가 데이터의 `answers.json`은 읽지 않습니다.

```text
dataset/test_public/prefix.json
dataset/test_public/<dialogue_id>/chunk_0.wav ...
models/whisper-small/
models/qwen2.5-0.5b/
```

저장소 루트에서:

```bash
# 모델과 수치 라이브러리 없이 실행 가능한 baseline
python solutions/day1_find_the_order.py --prefix-only

# 제공된 로컬 모델만 사용, 기본 540초의 소프트 예산
python solutions/day1_find_the_order.py --budget 540

# VRAM 부족 시 배치 축소, 화자 구분 영향 제거
python solutions/day1_find_the_order.py --asr-batch 2 --lm-batch 1 --no-speakers
```

전체 방법은 `numpy`, `scipy`, `torch`, `transformers`와 로컬 모델 파일이 필요합니다. 자동 다운로드나 설치는 하지 않습니다. 모델 로딩에 실패하면 stderr에 이유를 쓰고 유효한 prefix 답을 유지합니다. 모델을 전혀 돌리지 않는 경우에는 `--prefix-only`를 명시하는 편이 명확합니다.

Python에서 `main([...])`으로도 호출할 수 있습니다. 공식 제출용 독립 notebook에 소스를 포함할 때는 Jupyter의 자체 실행 인자를 넘기지 않도록 `main([])`을 호출해야 합니다.

## 시간·품질 조정

기본 예산의 55%까지 전사를 진행하고 나머지를 언어 모델에 배정합니다. 대화마다 점수 비중이 같으므로 짧은 대화부터 완성합니다. 전사가 끝나지 않은 대화나 연결 점수 계산이 중단된 대화는 prefix baseline을 유지합니다. `--asr-fraction`, `--asr-tokens`, `--target-tokens`, `--beam`으로 계산량을 조절할 수 있습니다.

시간 제한은 배치 사이에서 확인하는 **소프트 제한**입니다. 모델 로딩 또는 한 번의 추론 호출을 강제로 중단하지 않습니다. 기본 540초는 문제문 600초보다 여유를 둔 값이지만 모든 GPU에서 통과한다는 보장은 없습니다. 저장된 fallback도 프로세스가 강제 시간 초과되면 점수를 보장하지 않으므로 실제 환경에서 전체 실행 시간을 재야 합니다. 보존된 grader/CI의 시간 설정에는 문제문과 다른 값이 있어, 이 구현은 문제문의 제한을 기준으로 설계했습니다.

발화 하나는 기본적으로 처음 30초, 생성 토큰 128개까지만 사용합니다. 긴 발화는 끝부분을 잃을 수 있습니다. 화자 MFCC는 말 내용·잡음에도 영향을 받으므로 잘못된 교대 제약이 의심되면 `--no-speakers`와 비교하십시오.

인접 발화 자연스러움은 대회의 모든 쌍 순서 정확도와 같은 목적함수가 아닙니다. 더 발전시키려면 학습 대화로 `i가 j보다 먼저 나오는가`를 직접 학습하고, 각 발화의 예상 순위를 합산해 정렬하는 방법을 검토할 수 있습니다. 이것은 현재 코드에 포함하지 않은 대안이며, 더 좋은 성능이라고 단정할 근거는 없습니다.

공개 검증 데이터로 prefix baseline, 언어 모델만 사용, 화자 제약 포함의 세 점수를 비교한 뒤 선택하는 것이 좋습니다. 숨겨진 정답 조회, 미리 계산한 답 삽입, 허용되지 않은 모델 사용은 이 풀이에 포함되지 않습니다.
