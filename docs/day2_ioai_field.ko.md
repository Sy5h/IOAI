# Day 2 — IOAI Field

영역 분류와 값 예측을 분리한다. 첫 I의 1e10급 값, O의 나선, A의 -1, 마지막 I의 난수, 배경 0을 한 회귀 손실로 섞지 않는다. 이 구조는 공식 GAITE 힌트가 권장하는 방식이며 세부 학습 코드와 4-bit dropout은 독립 구현이다.

## 실행 환경과 제출

공식 `core`와 보호된 generator가 필요하다. 공개 원본 저장소의 runtime은 **Linux x86-64 CPython 3.13**에 맞춰져 있다. 일반 Windows Python이나 다른 ABI에서는 generator를 import할 수 없다. 이 저장소는 generator를 복제하거나 가짜 데이터로 교체하지 않는다.

공식 baseline 폴더에서 다음처럼 준비한다.

```text
solution.ipynb
custom_model.py              # solutions/field_custom_model.py의 내용
data/train_config/
    field_config.json        # 또는 여러 설정을 담은 field_configs.json
```

`solutions/field_custom_model.py`를 **`custom_model.py`라는 이름으로 복사**한다. `day2_ioai_field.py` 내용을 notebook 셀에 넣고 마지막 셀에서 `main([])`을 실행한다. CLI로는 다음과 같이 실행할 수 있다.

```bash
PYTHONPATH=../grading-original python /path/to/IOAI/solutions/day2_ioai_field.py --seconds 265
```

CLI 실행 위치에 `custom_model.py`가 있어야 최종 모델이 그 이름으로 직렬화된다. 로컬 개발에서는 `field_custom_model` fallback도 있지만, **공식 제출에는 `custom_model.py`와 `solution.ipynb` 두 파일을 제출**한다. model class를 notebook 안에서 다시 정의하면 별도 scorer가 복원하지 못할 수 있다.

결과 `model.pt`에는 전체 `nn.Module`을 저장한다. 설정이 여러 개면 모델 list를 저장한다. 채점기는 공개 설정 대신 숨겨진 설정을 같은 `data/train_config` 위치에 올리고 notebook을 다시 학습시킨다. 그래서 공개 JSON 값이나 공개 그림 좌표를 하드코딩하지 않는다.

## 쉬운 풀이 구성

| 구성 | 구현 |
|---|---|
| 영역 router | 2→64→64→64→5 MLP, 영역별 동일 수의 표본으로 분류 학습 |
| 첫 I | 값을 0~1로 정규화한 뒤 affine regression을 최소제곱으로 계산 |
| O | 2→64→64→64→1 MLP와 tanh 출력 |
| A | 영역을 맞히면 정확히 -1 |
| 마지막 I | 학습하지 않는 dropout + 선형층 |
| 배경 | 정확히 0 |

router는 `make_batch(..., include_regions=(영역명,))`로 얻은 표본에 cross-entropy를 적용한다. `argmax`에는 유용한 gradient가 없으므로 O 회귀와 분류는 별도로 학습한다. 첫 I가 실제로 선형 gradient인지 생성 표본의 정규화 MAE를 출력하며, 오차가 크면 경고한다. 해당 경우 작은 회귀망으로 바꾸는 추가 실험이 필요하다.

다른 영역의 점을 첫 I로 잘못 배정하면 그 영역 전체의 MAE 점수가 사라질 수 있다. 기본값은 영역당 16,384개 보정 표본에서 가장 큰 잘못된 I margin보다 1.5 높게 문턱을 잡는다. router 계산은 8,192점씩 처리하여 최대 margin만 누적한다. **이것은 표본 기반 보정이며 숨겨진 모든 점에서 거짓 양성이 없다는 보장은 아니다.**

## 마지막 I의 채점식에 맞춘 dropout

공식 scorer는 같은 점의 10회 출력에 대해 `numpy.std(..., ddof=0) / sqrt(1/12)`를 계산하고 1에서 자른다. 실제 값에 대한 MAE나 균등분포 적합도는 사용하지 않는다. 각 출력은 모두 [-2026,2026] 안이어야 한다.

`Dropout(0.5)`에 1 네 개를 넣고 `[128,256,512,1024]`로 가중합한 뒤 1920을 빼면 -1920부터 1920까지 간격 256인 16개 값이 나온다. `eval()`에서는 dropout이 꺼져 0이 된다. `train()`에서는 16개 값 중 하나가 나오며, 10번 중 두 값만 달라도 표준편차가 포화 기준보다 크다.

이상적인 독립 표본에서 10번 모두 같은 값일 확률은 `16^-9`다. 이 분석은 **마지막 I를 정확히 분류했을 때 그 가지의 점수**에만 해당한다. 전체 모델 점수나 경계 분류 성능을 의미하지 않는다. 4-bit 생성기의 범위와 실제 표준편차 식은 합성 테스트로 검증한다. 난수는 `nn.Dropout`에서만 나오며 추론 중 `torch.rand*`를 호출하지 않는다.

## 파라미터 제한과 시간

기본 모델은 고정 가중치를 포함해 **17,424개 파라미터**를 사용한다. 원본 scorer는 `>=20260`이면 점수를 절반으로 줄이므로 정확히 20,260개도 피한다. 모든 persistent tensor는 `nn.Parameter`로 등록하고 buffers를 사용하지 않는다. 모델 파일은 torch imports와 class 정의만 사용한다.

총 시간 기본값은 공식 5분보다 여유 있는 265초다. `--router-steps`, `--o-steps`로 각 학습 단계의 상한을 지정할 수 있다. 설정별 남은 시간을 나눠 쓰고 학습 단계도 deadline을 확인한다. 데이터 생성·최소제곱·저장 같은 단일 연산은 중간에 강제 종료하지 않으므로 실제 환경에서 시간 측정이 필요하다. 설정이 많을수록 각 모델의 학습 step이 줄어든다.

## 검증 범위와 남은 한계

합성 테스트는 dropout의 허용 범위·16개 출력·표준편차 점수, 파라미터 수, buffers 없음, torch-only imports, hard routing, A/배경 상수, 정규화 affine fit, 복수 config 읽기를 확인한다.

[첫 공식 runtime 실행](https://github.com/Sy5h/IOAI/actions/runs/38897670317)에서는 공개 설정의 각 영역 512점, 평가 seed 0으로 공식 baseline 37.0938점, 기존 저장소 모델 83.9189점을 측정했다. 모델 학습 subprocess는 각각 11.083/15.344초였다. Linux x86-64, CPython 3.13.16, PyTorch 2.8.0+cpu, CPU 4 threads 환경이며 패키지 설치·자료 다운로드·채점은 이 시간에서 제외한다. 기존 모델의 지역별 점수는 I 0.8433, O 0.6852, A 1, entropy 1, 배경 0.6675였다. 전체 모델에서도 dropout 방식이 동작했지만 영역 분류와 O 회귀에는 개선 여지가 있었다.

학습 step 증가도 실제로 시험했다. 공개 설정의 별도 검증 seed 1729·8071 평균 점수는 2,500/10,000/25,000 steps에서 각각 **85.6160/79.5103/78.7141**이었다. 오래 학습하는 방법은 개선되지 않아 기존 2,500 steps를 유지했다. 이 설정의 public/A/B 점수는 **83.9189/89.5119/68.1464**였고, 원본 저장소와 같았다. 각 실행의 실제 시간과 실패한 후보를 JSON에 보존한다.

B에서는 배경 점을 첫 I로 잘못 배정하는 한계가 드러났다. 큰 I 값을 배경 점에 출력하여 배경 점수가 0이 됐다. 이 결과를 본 뒤의 [보완 실험](https://github.com/Sy5h/IOAI/actions/runs/38900731329)에서 I 문턱의 표본 수와 safety를 늘렸다. 후보 `(영역당 보정 표본, safety) = (1024,0.5), (16384,1.5), (65536,2.5)`의 큰 public 검증셋 평균 점수는 각각 **70.5762/78.2887/77.1820**이었다. 검증셋은 seed 1729·8071, 영역당 8192점이며 보정 표본과 별도로 생성했다. 선택된 기본값은 **16,384 / 1.5**다.

최종 재측정은 원래 공식 평가 설정을 사용했다. 모든 설정의 점수가 오르지는 않았다.

| 공개된 설정 | 기존 1024 / 0.5 | 최종 16384 / 1.5 | 최종 학습 시간 |
|---|---:|---:|---:|
| public | 83.9189 | 77.5777 | 26.272초 |
| leaderboard A | 89.5119 | 85.4997 | 20.764초 |
| leaderboard B | 68.1464 | 72.7411 | 24.920초 |

큰 생성 검증셋에서의 안정성과 가장 낮은 공개 split의 개선을 기준으로 기본값을 정했다. A는 4.01점 떨어지고 B는 4.59점 올랐다. 원본 문턱을 재현하려면 `--validation-per-region 1024 --i-safety 0.5`를 사용한다. **A/B 결과를 본 뒤 동기를 얻은 후속 실험이므로 그 재측정을 독립적인 확증으로 주장하지 않는다.** 파라미터 수는 두 방법 모두 17,424개다.

A/B 경로명은 `private`지만 실제로 공개된 JSON만 사용한다. 공개 파일은 A/B 각각 단일 설정으로, B가 복수 설정이라는 grader README 설명과 다르다.

[실측 JSON](../benchmarks/results_vision.json)과 [재현 명령](../benchmarks/vision_reproduction.ko.md)에 모든 측정과 provenance를 남긴다. 이것은 공개된 설정의 CPU 재현 결과이며 미공개 대회 리더보드 제출 결과가 아니다. 공식 소개의 baseline 31.2187, 위원회 63.53도 별도 수치로 구분한다.

출처: [공식 문제](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/6_IOAI_Field/statement.md), [공식 GAITE 힌트](https://github.com/IOAI-official/IOAI-2026/blob/main/GAITE-Contest/6_IOAI_Field/hint.md), [실제 scorer](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/6_IOAI_Field/code/grading-original/metrics/field_score.py), [실행·직렬화 계약](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/6_IOAI_Field/code/grading-original/README.md).
