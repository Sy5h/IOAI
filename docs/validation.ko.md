# 검증한 범위와 남은 한계

검증일: 2026-10-15. **실제 공식 데이터 전체의 점수와 실행 시간은 아직 측정하지 않았습니다.** 이 문서의 테스트 통과는 입출력·알고리즘·포장 동작의 확인이며, 리더보드 점수 보장이 아닙니다.

## 수행한 검사

```bash
python tools/build_notebooks.py
python -m unittest discover -s tests -v
python tools/build_notebooks.py --check
python -m compileall -q solutions tools tests
```

전체 합성·통합·포장 테스트 **40개 통과**. 통합 실행의 unittest 측정 시간은 약 17.4초였습니다. 공식 데이터 학습 시간과 비교할 수 있는 수치가 아닙니다.

| 범위 | 테스트 수 | 확인한 내용 |
|---|---:|---|
| Find the Order / Robot Chasing | 13 | 역순열, prefix 유지, 순열 유효성, 화자 제약 fallback, Qwen target-token mask, 명령 파싱, 막힌 이동의 방향 의미, 행 순서, 작은 학습 smoke test |
| Double Agent Dilemma / IOAI Field | 9 | 차등 공격·축소, clipping 입력 동등성, 실패/시간 초과 영텐서, ZIP 텐서 크기, dropout 범위·평가식, 파라미터 수, hard routing, affine fit, 복수 config |
| Potato / Ghost | 14 | 관측 방향·동점·게임 reset, 초기 두 단어 후보 유지, 작은 게임 시뮬레이션, 원문 offset, 글 단위 holdout, 단일 경계, 문장 구분점 없는 글, 시간 부족 fallback, 통제된 임베딩 통합 검사 |
| 노트북 포장 | 4 | 소스 일치, 문법, 파일 크기, 실행 한 번, Potato의 Jupyter/nbconvert 분기, Field 보조 파일 일치 |

## 독립 검토에서 수정한 문제

Ghost에서 중간 문장 구분점이 없는 글을 받으면 후보가 0 또는 글 끝뿐이 되는 문제를 수정했습니다. 이제 가능한 경우 양쪽 최소 180단어 조건을 만족하는 원문의 단어 시작 위치를 사용합니다. 일반 추론과 시간 부족 fallback을 모두 회귀 테스트로 확인했습니다.

노트북 생성기는 Python 파일의 최상위 `__main__` 실행부를 제거한 뒤 실행 셀을 한 번만 넣습니다. 따라서 같은 코드를 붙여 넣고 `main([])`을 추가해 두 번 학습하는 문제를 방지합니다.

## IOAI Field의 변동성 출력

현재 모델은 **17,424개 파라미터**이며 등록 buffer가 없습니다. 최종 I의 dropout 출력은 -1920부터 1920까지 16개 값입니다. 합성 좌표 5,000개에 각 10회 출력하여 공식 표준편차 식으로 확인한 영역 점수는 1.0이었습니다. 같은 4-bit 식의 별도 100,000개 좌표 실험에서도 범위 위반이 없고 평균 1.0이었습니다.

이는 마지막 I 출력 부분만의 실험입니다. 실제 로고의 영역을 맞히는 능력, O의 회귀 성능, 숨겨진 설정에 대한 전체 Field 점수를 측정한 결과는 아닙니다.

## 실행하지 않은 부분

- 공식 음성 데이터에서 Whisper·Qwen 전체 추론 및 600초 제한 검증.
- 실제 60,000행 Robot 학습과 공식 평가 세트 점수.
- Potato의 실제 비공개 표현을 사용한 120게임 평가.
- 실제 ImageNet 이미지·제공된 두 모델에 대한 전체 200개 공격.
- 실제 BGE 가중치·공식 글 데이터의 학습과 경계 점수.
- Field의 Linux CPython 3.13 보호 런타임에서 숨겨진 설정 재학습 및 공식 scorer 실행.

추가 평가 시에는 공개 데이터를 개발에 사용하고, 평가할 별도 분할을 정해 성능을 기록하세요. 코드의 기본 하이퍼파라미터를 최적값으로 주장하지 않습니다.

## 이번 검증 환경

- Python 3.11.0, Windows.
- NumPy 2.3.3, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121.
- transformers 4.45.0, SciPy 1.16.2, scikit-learn 1.7.2, Pillow 11.3.0.
- 실제 공격 모델 로드에 필요한 timm은 이 검증 환경에 설치되지 않았습니다. 합성 공격 테스트는 작은 대체 모델로 수행했습니다.

환경 버전은 결과를 해석하기 위한 기록입니다. 모든 버전 조합이나 GPU에서 같은 시간·점수를 보장하지 않습니다.

