# Field 두 번째 개선 실험 재현

이 실험의 비교 기준은 저장소 commit `991a49b48092286967b5bd7cada465e336ebc4af`다. 기존 실험과 점수는 `results_vision.json`에 그대로 보존한다. 이전 A/B 결과에서 드러난 문제를 보완하는 개발 재평가이며, 새로운 독립 테스트나 미공개 리더보드 제출로 해석하지 않는다.

[실행 38907040090](https://github.com/Sy5h/IOAI/actions/runs/38907040090)은 성공했다. public 생성 holdout 평균은 기준 78.4457점, 선택된 **PCA 32띠 86.5326점**이었다. 고정한 선택안의 최종 public/A/B 점수는 **85.1823/91.5164/81.8590**, 같은 실행에서 다시 측정한 기준은 **77.5777/85.4997/72.7411**이었다. 전체 후보, 명령, 시간, 하드웨어, 소스 SHA는 [실측 JSON](round2_field_results.json), 실행 원문은 [경로를 정규화한 로그](round2_field.log)에 있다.

## 실행

보호된 공식 generator가 지원하는 Linux x86-64 CPython 3.13에서 실행한다. Windows에서 generator를 대체하거나 보호 runtime을 변경하지 않는다.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.3 matplotlib==3.10.6
python tools/benchmark_field_round2.py --work-dir work/field-round2
```

GitHub Actions의 **IOAI Field round 2 CPU benchmark**도 같은 명령을 사용한다. 수동 실행만 가능하고 표준 `ubuntu-latest` CPU runner, 4 threads, 최대 30분이며 secrets를 사용하지 않는다. 설치·다운로드 시간과 학습·채점 시간을 구분하여 기록한다.

공식 코드 revision은 `e1963801aa4cf33a5a03520b39fa6620a7894c54`, HF 데이터 revision은 `35136ba0abb05ea8dcdb6ec3db200617a9da6104`다. 필요한 공개 JSON과 공식 runtime만 작업 폴더에 내려받고 파일별 SHA-256을 기록한다. 데이터·정답·모델 가중치는 저장소에 추가하지 않는다.

## 비교와 선택

1. 기존 margin router와 새 표본 경계 네 후보를 비교한다. 새 후보는 16/32개 띠와 affine/PCA 축의 조합이다.
2. 후보는 public config에서만 학습한다. `make_batch`가 제공하는 I 좌표와 비-I 좌표 외에 보호된 구현이나 실제 경계 정보를 읽지 않는다.
3. 후보 선택용 점수는 public config의 별도 생성 seed `314159`, `271828`, 영역당 8,192점에서 공식 `evaluate_model`로 측정한다. 두 점수 평균이 가장 높은 설정을 선택하며 동률이면 적은 띠 수를 우선한다.
4. 선택 뒤 설정을 고정하고 기준 코드와 함께 public/A/B 각 config에서 새로 학습한다. 최종 채점은 각 config에 공개된 원래 `eval_config.json`을 사용한다. 실제 공개된 A/B는 각각 단일 field config다.

새 경계는 I 표본을 한 축으로 정렬한 다음 같은 수의 표본을 담는 띠로 나누고, 각 띠의 볼록껍질 안쪽만 사용한다. affine 축은 I의 학습된 값 증가 방향, PCA 축은 I 표본이 가장 길게 퍼진 방향이다. 비-I 보정 표본이 안으로 들어가는 조각은 통째로 버린다. 첫 I만 이 경계로 판정하고 나머지 네 영역은 기존 MLP가 분류한다.

이는 표본 기반 추정이다. 볼록껍질이 보지 못한 오목한 부분을 메우거나 표본 사이의 빈 공간에서 오류를 낼 가능성이 남는다. 검증 표본에 거짓 양성이 없다고 해서 모든 숨은 점에서도 없다고 보장하지 않는다.

기본 MLP는 17,424개 파라미터다. 새 경계는 띠당 최대 20개 선형 부등식, 부등식당 3개 파라미터를 추가하므로 32개 띠의 최대값은 19,344개다. 모든 계수는 고정 `nn.Parameter`로 등록되고 버퍼를 쓰지 않는다. 추론은 torch만 사용한다.

원시 결과는 `work/field-round2/summary.json`, 파일 provenance는 `provenance.json`, 각 학습/채점 로그는 실행별 하위 폴더에 생긴다. stdout의 `ROUND2_SELECTION`, `ROUND2_SELECTED`, `ROUND2_FINAL_RESULT` 줄에도 완전한 실행별 JSON을 남긴다. 공개용 결과를 정리할 때 runner 절대 경로는 `<repository>` 또는 `<work-dir>`로 정규화한다.

## 현재 기본값과 측정 코드의 관계

측정은 `a5053a9b20d18eacc590590bd1cb5d6a473cb747`에서 다음 두 옵션을 명시해서 수행했다.

```bash
--i-geometry-bands 32 --i-geometry-axis pca
```

결과 확인 뒤 바꾼 것은 `train_one_config`와 CLI의 기본값 선언 세 곳뿐이다. 그 선언을 되돌린 소스의 SHA-256은 측정 당시 `04ea837608021c1015ac0d1ee859883e9c0a4cb17b417828df08358ba74828dd`와 정확히 일치한다. 최종 학습 소스 SHA-256은 `1a995862c3b5a2e6da7760de42904e6b92a13fc656fb542906b202aebc720267`이다. 모델 소스는 측정 전후 동일하며 SHA-256은 `c76c6f113ac5012f8911b1543f0a6b112d9da9b73570eee5d1252d1fc08d4b5c`다.

기본 CLI와 명시적 선택 옵션이 같은 훈련 인자를 전달하는 테스트를 포함하여 관련 합성/계약 테스트 16개가 통과했다. 기본값 변경 뒤 전체 공식 학습을 중복 실행하지 않았다. 이전 margin 방식을 재현하려면 `--i-geometry-bands 0 --validation-per-region 16384 --i-safety 1.5`를 지정한다.
