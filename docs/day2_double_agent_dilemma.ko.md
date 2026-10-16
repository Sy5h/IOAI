# Day 2 — Double Agent Dilemma

이 코드는 공식 모델 두 개를 그대로 두고 이미지 픽셀 변화량만 최적화한다. A는 ResNet 정답·ViT 오답, B는 ViT 정답·ResNet 오답을 만든다. 테스트 정답 라벨을 사용하는 것은 문제에서 명시적으로 허용한다.

## 실행

공식 환경에는 다음 파일이 준비되어 있어야 한다. 새 pretrained 모델을 내려받는 코드는 없다.

```text
models/resnet18.pth
models/vit_tiny_patch16_224.safetensors
dataset/test_public/labels.json
dataset/test_public/images/0000.png ...
```

CLI에서는 공식 baseline 작업 폴더를 현재 디렉터리로 삼아 실행한다.

```bash
python /path/to/IOAI/solutions/day2_double_agent_dilemma.py --seconds 660
```

공식 제출용 `solution.ipynb`에는 이 Python 파일의 내용을 한 셀에 넣고, 마지막 셀에서 `main([])`을 실행한다. notebook 환경에서는 자동으로 CLI 인수를 파싱하지 않는다. `DATA_DIR`, `MODELS_DIR`, `TEST_SPLIT` 환경변수를 존중하므로 숨겨진 split 경로를 고정하지 않는다.

결과는 `submission.zip`이다. 최상위에 `{index}_a.pt`, `{index}_b.pt`가 들어가며 모든 tensor는 **원본 이미지 해상도의 3×H×W**이다. 공식 제출물은 ZIP 자체가 아니라 이를 생성하는 `solution.ipynb`이다. 공식 notebook의 기존 naive baseline 실행 셀이 이 결과를 다시 덮어쓰지 않게 교체한다.

## 알고리즘

1. 모델을 평가 모드로 고정하고 가중치 gradient를 끈다. 입력 gradient는 유지한다.
2. `delta=0`에서 시작해 학습률 0.0001의 Adam으로 변화량을 찾는다. 손실은 틀릴 모델의 정답 logit margin을 내리고, 유지할 모델의 margin을 보호하며 L2 크기를 벌점으로 준다.
3. 성공한 순간 변화량을 저장하고, 더 작은 배율을 여러 번 시험한다. 축소 중 실패해도 직전 성공 후보를 보존한다.
4. 작은 학습률 시도에는 이미지별 시간의 65%를 배정한다. 실패하면 예약한 나머지 시간을 사용해 기존 학습률 0.001로 한 번 재시도한다. 끝까지 못 푼 이미지에는 영텐서를 기록한다.

`Resize(256) → CenterCrop(224) → ImageNet Normalize`를 tensor 연산으로 수행한다. `Resize(256)`은 짧은 변을 256으로 맞추며 비율을 유지한다. 224×224에서 공격한 뒤 원본 크기로 대충 늘리는 방식은 사용하지 않는다.

## 채점식을 이용하는 간단한 개선

- **Clipping 정리:** 저장 전 `clamp(image+delta,0,1)-image`로 교체한다. 실제 입력은 같고 쓸모없는 변화량은 줄어든다.
- **실패한 변화량은 0:** 실패는 성공 횟수를 늘리지 않으면서 전체 L2 벌점에는 포함된다. 영텐서로 바꾸면 이 손해가 없어진다.
- **성공 뒤 축소:** 더 큰 오답 확률은 추가 점수를 주지 않는다. 성공 조건을 유지하는 가장 작은 후보가 유리하다. 배율에 따른 성공 여부가 단조라는 보장은 없으므로 축소는 최적성 보장이 없는 탐색이다.

실제 scorer는 성공 비율에 아래 penalty factor를 곱한다.

```text
m = mean_i((norm(delta_A) + norm(delta_B)) / (3*H_i*W_i))
PF = 1.5 - sigmoid(50000*m)
score = (successful_A + successful_B)/(2*N) * PF
```

기본 epsilon 0.004는 공식 제약이 아니라 탐색 시작값이다. 20 step마다 상한을 키운다. `--max-steps`, `--epsilon`, `--learning-rate`는 공개 train split으로 조절한다. 시간 할당은 GPU 연산, 이미지 읽기, ZIP 쓰기 속도에 따라 달라지므로 공식 12분보다 여유 있게 660초를 기본값으로 둔다. 한 연산의 실행 도중까지 강제로 중단하는 제한은 아니다.

## 검증 범위와 남은 한계

합성 분류기에서 차등 공격 성공, 성공 유지 축소, 작은 학습률 실패의 fallback 복구, 불가능한 공격의 영텐서 처리, clipping 동등성, flat ZIP round-trip을 테스트한다. 실제 ResNet·ViT와 수정하지 않은 공식 scorer로도 측정했다.

공개 train 첫 20장에서 학습률 0.001/0.0003/0.0001은 각각 97.5271/97.7515/97.8512점이었다. 모두 A/B 합계 40개 공격이 성공하여 0.0001을 선택했다. 그러나 public 100장에서는 원본 97.1192점·200/200 성공에 비해 작은 학습률 단독은 95.7225점·196/200 성공으로 나빠졌다. **실패한 실험을 그대로 기록하고**, 이 결과를 참고하여 실패 시 0.001로 재시도하는 방법을 추가했다. 따라서 public 100장은 최종안의 독립 검증셋으로 주장하지 않는다.

이후 처음 평가한 공개 leaderboard A/B 각 100장에서 예약 없는 fallback은 A **97.5371점**, B **96.9422점**이었다. 원본은 각각 **97.1896/97.0480점**이다. A에서는 200/200개 공격이 성공했지만 B에서는 199/200개만 성공했다. 실패한 첫 시도가 이미지별 시간을 전부 쓰면 재시도할 시간이 없다는 코드 경로를 확인했다. 이것이 해당 실패의 원인이었다고 확정할 시도별 시간 로그는 없다.

최종안에는 첫 시도 65%·재시도 예약 35%라는 일반적인 예산 분배 보완을 한 번 추가했다. A/B를 이미 본 뒤의 수정이므로 최종 재측정은 개발 검증이며 독립적인 확증으로 주장하지 않는다. `--first-attempt-fraction 1`로 예약 없는 이전 정책을 재현할 수 있다. 다운로드 경로 이름이 `private`여도 이 revision에서 이미 공개된 데이터이며 실제 미공개 리더보드 점수가 아니다. [실측 JSON](../benchmarks/results_vision.json)과 [재현 명령](../benchmarks/vision_reproduction.ko.md)에 데이터 revision, 전체 점수, 시간, 하드웨어, 실패 실험을 함께 남긴다.

| 공개된 설정 | 원본 점수 | 최종 점수 | 원본 실행 | 최종 실행 | 최종 공격 성공 |
|---|---:|---:|---:|---:|---:|
| leaderboard A, 100장 | 97.1896 | **97.5372** | 70.796초 | 154.954초 | 200/200 |
| leaderboard B, 100장 | 97.0480 | **97.3802** | 69.235초 | 164.672초 | 200/200 |

A/B 점수는 각각 0.3476/0.3321점 올랐고 공격 400개가 모두 성공했다. 실행 시간은 각각 약 **2.19배/2.38배**로 늘었다. 모두 660초 예산 안에서 끝났으며, 마지막 예약 수정 이후에는 추가 튜닝을 하지 않았다. **최종 코드를 `test_public` 100장에 다시 실행하지는 않았다.** 앞의 public 수치는 원본과 작은 학습률 단독 실험의 결과다.

공개100의 원본/작은 학습률 단독 실행 시간은 각각 77.235/159.547초였다. RTX 4070 Laptop 8 GB에서 새 Python 프로세스의 imports·CUDA 초기화·모델 로딩·공격·ZIP 생성을 포함하며, 다운로드·설치·별도 채점은 제외한다. 공식 제한은 다른 GPU/실행 환경이므로 이 시간만으로 대회 환경의 시간을 보장하지 않는다. 위원회 점수 96.99는 이 코드의 측정값이 아니다.

출처: [공식 문제](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/4_Double_Agent_Dilemma/statement.md), [원본 채점 코드](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/4_Double_Agent_Dilemma/code/grading-original/evaluate.py), [공식 GAITE 힌트](https://github.com/IOAI-official/IOAI-2026/blob/main/GAITE-Contest/4_Double_Agent_Dilemma/hint.md).
