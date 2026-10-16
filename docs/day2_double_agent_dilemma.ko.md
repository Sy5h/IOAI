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
2. `delta=0`에서 시작해 Adam으로 변화량을 찾는다. 손실은 틀릴 모델의 정답 logit margin을 내리고, 유지할 모델의 margin을 보호하며 L2 크기를 벌점으로 준다.
3. 성공한 순간 변화량을 저장하고, 더 작은 배율을 여러 번 시험한다. 축소 중 실패해도 직전 성공 후보를 보존한다.
4. 각 공격에 남은 시간의 몫을 배정한다. 못 푼 이미지에는 영텐서를 기록해 출력 형식을 완성한다.

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

합성 분류기에서 차등 공격 성공, 성공 유지 축소, 불가능한 공격의 영텐서 처리, clipping 동등성, flat ZIP round-trip을 테스트한다. 실제 ResNet·ViT의 전체 공개/비공개 데이터 공격은 이 저장소 작성 과정에서 실행하지 않았다. 위원회 점수 96.99는 이 코드의 성능이 아니다. 성공률과 penalty를 반드시 공식 scorer로 확인해야 한다.

출처: [공식 문제](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/4_Double_Agent_Dilemma/statement.md), [원본 채점 코드](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/4_Double_Agent_Dilemma/code/grading-original/evaluate.py), [공식 GAITE 힌트](https://github.com/IOAI-official/IOAI-2026/blob/main/GAITE-Contest/4_Double_Agent_Dilemma/hint.md).
