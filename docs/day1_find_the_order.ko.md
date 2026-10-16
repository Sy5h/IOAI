# Day 1 · Find the Order

공식 공개 데이터로 평가하는 제안 풀이입니다. 공식 정답 코드나 입상자의 검증된 최고 성능 풀이가 아닙니다. 출력 형식과 핵심 계산은 합성 테스트로 검사하고, 실제 데이터 점수와 시간은 아래 실측 표에 별도로 기록합니다. 원대회 Linux 컨테이너가 아닌 로컬 Windows 환경의 결과입니다.

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
5. 첫 두 발화를 고정한 beam search로 연결 점수 합이 큰 순서를 찾습니다. 기본 beam 크기는 256입니다. 2차 공개 개발 실험에서 순서 후보를 더 오래 남기는 이 한 가지 설정만 바꿨습니다.
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

# 이 저장소에서 실행하는 CPU 음향 대안: scipy/numpy 필요, 모델 가중치 불필요
python benchmarks/order_acoustic_candidate.py --test-dir dataset/test_public --output answers.json

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

## 1차 공개 데이터 실측: beam 16 기준선

데이터 revision은 `71a35b5a566665e242416eba45f018dc378663b8`입니다. `public/pretrain` 20대화만 개발용으로 사용하고, 최종 평가 정답은 별도 scorer에서만 읽었습니다. `public/pretest` 20대화는 원래 공개 시험의 일부입니다. 현재 공개된 대회 후 자료인 A/B는 각각 100대화입니다.

| 방법 / 평가 구간 | 대화별 쌍 순서 정확도 | 새 프로세스 벽시계 |
|---|---:|---:|
| Prefix-only / pretrain 개발용 20개 | 69.3335% | 0.269초 |
| Prefix + 화자 교대만 / pretrain 개발용 20개 | 70.5481% | 6.278초 |
| 원본 Whisper + Qwen / pretrain 개발용 20개 | **75.5781%** | 109.517초 |
| Prefix-only / pretest 20개 | 68.9534% | 0.263초 |
| 원본 Whisper + Qwen / pretest 20개 | **75.9140%** | 82.715초 |
| Prefix-only / Leaderboard A 100개 | 68.9690% | 0.551초 |
| Prefix-only / Leaderboard B 100개 | 69.1460% | 0.476초 |
| Prefix + 화자 교대만 / Leaderboard A 100개 | 69.0880% | 42.270초 |
| Prefix + 화자 교대만 / Leaderboard B 100개 | 69.7803% | 43.236초 |
| 원본 Whisper + Qwen / Leaderboard A 100개 | **76.5214%** | 362.131초 |
| 원본 Whisper + Qwen / Leaderboard B 100개 | **74.0884%** | 266.970초 |

Prefix-only와 화자 교대 후보는 CPU에서 Whisper와 Qwen 없이 실행했습니다. 원본 Whisper + Qwen 개발 실행은 RTX 4070 Laptop GPU 8GB에서 20개 대화 전부를 처리했고 오류·fallback은 없었습니다. 원본 설정을 변경하지 않았습니다. 모델 revision은 Whisper-small `973afd24965f72e36ca33b3055d56a652f456b4d`, Qwen2.5-0.5B `060db6499f32faf8b98477b0a26969ef7d8b9987`입니다.

같은 설정의 전체 GPU 평가도 A/B 각각 **100/100개 대화를 Qwen으로 복원했고 prefix fallback은 0개**였습니다. `ordered dialogue` 로그 개수를 직접 확인했으며, 각 출력의 잘못된 순열도 0개였습니다. Prefix-only보다 A에서 +7.5524%p, B에서 +4.9425%p 높았습니다. 원본 전체 파이프라인의 유효성을 확인한 결과이며, 이번 실측을 통해 그 소스를 개선했다는 뜻은 아닙니다. A/B 실행 사이에도 소스와 하이퍼파라미터를 바꾸지 않았습니다.

Pretest도 20개 대화 전부 처리했습니다. `public/pretest/11/chunk_2.wav`는 실제 249,856바이트인데 RIFF 헤더가 274,430바이트를 선언하는 원본 파일 문제로 경고가 났습니다. 고정 revision URL에서 별도로 받은 원본과 평가용 파일의 SHA256이 `adc5d9d2672718da8c8284578e5d1700231abbc9af42922447cf8bd51dda1784`로 일치함을 확인했습니다. 데이터를 수정하지 않고 scipy가 읽은 실제 오디오 부분을 그대로 사용했습니다. 경고 때문에 fallback으로 전환된 것은 아닙니다.

화자 교대만 사용하는 작은 후보는 개발용 20개 중 15개 대화에서 보수적 화자 판정이 통과했습니다. 이때 채택된 157개 클립의 화자 구분은 개발 정답의 교대 순서와 모두 일치했습니다. 이 진단은 순서 정확도와 다른 수치입니다.

설정을 고정하고 평가한 A/B에서는 각각 72개, 71개 대화에 화자 제약이 적용됐고 prefix-only 대비 +0.1190%p, +0.6344%p였습니다. 상승 폭은 작습니다. GPU나 모델 가중치 없이 실행하는 대안으로 볼 수 있지만, 이것만으로 대화 내용의 순서를 충분히 복원하지는 못합니다. A/B를 본 뒤 설정을 추가 조정하지 않았습니다.

각 설정 1회, CPU Intel Core Ultra 9 185H / threads 4 / Python 3.11 환경입니다. 시간에는 라이브러리 import와 입력 읽기를 포함하고, 다운로드·scorer는 제외했습니다. 모델 가중치가 없어도 prefix-only 수치는 재현할 수 있습니다. [실측 JSON과 전체 명령](../benchmarks/results_order_robot.json), [실행·채점 재현기](../benchmarks/reproduce_order_robot.py), [고정 revision 다운로드 도구](../benchmarks/download_order_robot.py), [음향 후보 코드](../benchmarks/order_acoustic_candidate.py)를 참고하십시오.

저장된 예측에는 고정 revision의 공식 `score_submission`도 별도로 적용했습니다. [교차 검증 결과](../benchmarks/scoring_audit_order_robot.json)는 이 문제 11회와 Robot Chasing 8회, 총 19회 모두 점수가 소수점 4자리까지 일치함을 확인합니다. 이는 채점 수식의 확인이며 원대회 Podman·notebook·시간 제한 환경 전체를 재현했다는 뜻은 아닙니다.

## 2차 개선: beam 후보 16개에서 256개로

복잡한 모델이나 새 학습을 추가하지 않고 **기본 `--beam`을 16에서 256으로 바꿨습니다.** 처음에 좋아 보이는 순서가 뒤에서 막힐 때를 대비해 후보 경로를 더 오래 남깁니다. 인접 발화 점수 자체와 화자 판정은 그대로이며 더 넓은 탐색이 모든 대화의 정답률을 높인다는 보장은 없습니다. `--beam 16`으로 기존 알고리즘을 다시 실행할 수 있습니다.

설정 선택에는 `public/pretrain` 20대화만 사용했습니다. Whisper 전사와 Qwen 연결 점수를 한 번 계산한 뒤 CPU에서 미리 정한 8개 대안을 비교했습니다. beam 16/64/256, 화자 조건 제거, 목적함수 중심화·전역 순서 항을 비교했고 평균이 가장 높은 beam 256 하나만 채택했습니다.

| 동일 개발 캐시의 탐색 | 순서 정확도 | 캐시 이후 CPU 탐색 시간 |
|---|---:|---:|
| beam 16 | 75.8797% | 0.0101초 |
| beam 64 | 79.5784% | 0.0361초 |
| **beam 256** | **80.0778%** | **0.1219초** |

이 시간은 20대화의 **순서 탐색만** 포함하며 전체 풀이 시간이 아닙니다. 개발 캐시는 LM batch 8과 모든 사용 가능한 연결을 계산했고 최종 실행은 기존 LM batch 4를 유지합니다. 따라서 캐시 기준선 75.8797은 1차 전체 실행의 75.5781과 별도 결과입니다. 개발 기록의 첫 10개/뒤 10개도 설정 선택에 쓴 같은 자료이며 독립 holdout이 아닙니다. 새 최종 실행에는 이 캐시를 전달하지 않고 전사부터 다시 계산합니다.

기준 commit은 `991a49b48092286967b5bd7cada465e336ebc4af`입니다. 그 소스의 SHA256이 1차 실측 기준선과 정확히 같음을 확인했습니다. 아래 기준선의 점수·시간은 **과거 1차 실행 기록**이며 이번에 나란히 다시 돌린 기준선 시간이 아닙니다. 이번 후보는 별도의 새 프로세스로 실행했습니다.

| 공개 평가 | 기존 beam 16 / 과거 시간 | 새 beam 256 / 이번 시간 |
|---|---:|---:|
| pretest 20대화 | 75.9140% / 82.715초 | **79.1531% / 78.749초** |
| Leaderboard A 100대화 | 76.5214% / 362.131초 | **77.7727% / 277.682초** |
| Leaderboard B 100대화 | 74.0884% / 266.970초 | **77.9223% / 254.286초** |

pretest +3.2390%p, A +1.2513%p, B +3.8339%p였습니다. 이번 고정안은 세 분할 모두 개선되어 기본 beam 256을 채택했습니다. 후보 설정을 최종 A/B 결과에 맞춰 추가로 조절하지 않았습니다.

A가 빨라진 것을 beam 변경의 속도 효과라고 해석하면 안 됩니다. 후보 수를 늘리면 탐색 계산은 증가합니다. 서로 다른 실행 시점의 OS·GPU 캐시와 동시 작업 상태가 같지 않으므로 정확도와 별도로 실행 시간을 읽어야 합니다. 다운로드·scorer를 제외하고 import, 입력·모델 읽기, 전사·언어 모델 추론, 답 저장·종료를 포함합니다.

처음 pretest 명령은 존재하지 않는 입력 폴더를 지정해 약 0.254초에 `FileNotFoundError`로 끝났습니다. 이 실패에는 점수가 없습니다. 실제 다운로더 경로로 고친 실행은 20대화 모두 완료했습니다. A/B도 각각 100대화 모두 Qwen 순서 복원을 끝냈고 fallback과 잘못된 순열은 없었습니다. 과거 기준선 3회와 새 후보 3회 모두 고정 revision의 공식 `score_submission` 결과와 소수점 4자리까지 일치했습니다.

이미 과거 A/B 결과를 관찰한 뒤의 개발입니다. 이번 설정은 새 pretest/A/B 실행 전에 고정했고, 이후 평가 결과로 재조정하지 않았습니다. 따라서 **공개된 자료의 개발 재평가**이며 새 비공개 데이터에 대한 독립 확증은 아닙니다. [2차 결과 JSON](../benchmarks/round2_order_results.json), [실행 로그](../benchmarks/round2_order_runs.log), [재현 절차](../benchmarks/round2_order_reproduction.ko.md)에 선택 과정과 성공·실패 기록을 보존합니다. 정답 배열·오디오·전사문·개발 캐시·가중치는 게시하지 않습니다.

## 일반 clone에서 재현

아래는 저장소 최상위에서 실행하는 PowerShell 예시입니다. 앞서 설명한 Python 의존성과 `huggingface_hub`가 설치되어 있어야 합니다. 다운로드는 고정 revision을 사용하며 공개 오디오 전체와 두 모델에 약 2.71GB가 필요합니다. 실제 실행 기록 JSON의 `<workspace>/outputs/IOAI` 경로를 복사할 필요는 없습니다.

```powershell
python benchmarks/download_order_robot.py order-data --workspace work/bench-order-robot
python benchmarks/download_order_robot.py order-eval --workspace work/bench-order-robot
python benchmarks/download_order_robot.py models --workspace work/bench-order-robot
$benchPath = (Resolve-Path work/bench-order-robot).Path
python benchmarks/reproduce_order_robot.py run --task order --label full_a --script solutions/day1_find_the_order.py --dataset work/bench-order-robot/order/test_leaderboard_a/dataset --truth work/bench-order-robot/scorer-only/order/test_leaderboard_a_answers.json --run-dir work/bench-order-robot/runs/full_a --data-revision 71a35b5a566665e242416eba45f018dc378663b8 --split test_leaderboard_a --device cuda --threads 4 --timeout 630 -- --budget 540 --whisper "$benchPath/models/whisper-small" --qwen "$benchPath/models/qwen2.5-0.5b"
```

`order-data`를 먼저 실행해야 `order-eval`의 공통 데이터 배치가 완료됩니다. B는 명령의 `test_leaderboard_a`와 `full_a`를 각각 `test_leaderboard_b`, `full_b`로 바꿉니다. 이미 예측 파일이 있는 실행 폴더는 재사용하지 않습니다. `--truth`는 재현 실행기의 사후 scorer에만 전달되며, 풀이 프로세스에는 입력 디렉터리와 모델 경로만 전달됩니다. Linux에서는 `$benchPath` 대신 `"$(pwd)/work/bench-order-robot"`를 모델 경로 앞에 사용하면 됩니다.
