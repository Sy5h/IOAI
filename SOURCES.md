# 공식 자료와 출처

확인 기준: 2026-10-16. 온라인 원본은 이후 갱신될 수 있습니다. 실제 실행은 아래 revision에 고정했습니다.

- [IOAI 2026 공식 저장소](https://github.com/IOAI-official/IOAI-2026)
- [본선 문제·환경 목록](https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest)
- [공식 GAITE 힌트](https://github.com/IOAI-official/IOAI-2026/tree/main/GAITE-Contest)
- [사전 과제 원본 노트북](https://github.com/IOAI-official/IOAI-2026/tree/main/At-Home-Round)
- [대회 규정](https://ioai-official.org/wp-content/uploads/2026/05/IOAI2026-Contest-Rules.docx-1.pdf)

각 문제 폴더의 `statement.md`, `code/baseline-original/`, `code/grading-original/`와 해당 GAITE `hint.md`를 기준으로 구현·해설을 작성했습니다. 각 한글 풀이에 직접 출처 링크가 있습니다. 교육용 `baseline/`, `grading/`과 보존된 대회 원본이 다를 때 원본을 우선했습니다.

## 원자료 저작자

문제, 문제문, 원본 코드와 힌트의 저작·개발 기여는 IOAI 2026의 Host/International Scientific Committees 및 각 문제 출제자에게 있습니다. [전체 기여자 명단](https://github.com/IOAI-official/IOAI-2026/blob/main/Individual-Contest/README.md#credits)을 참고하세요.

공식 저장소는 [CC BY 4.0](https://github.com/IOAI-official/IOAI-2026/blob/main/LICENSE)로 자료를 공개했습니다. 이 저장소는 공식 저장소의 배포본이 아니며, 한국어 설명과 별도의 제안 구현을 추가한 학습 자료입니다. 공식 로고·원문 전체·데이터·모델 가중치를 재배포하지 않습니다.

## 구현과 공식 자료의 관계

- Find the Order: 공식 힌트의 전사·화자 구분을 사용하며, Qwen 조건부 점수와 작은 beam search는 제안 구현입니다.
- Robot Chasing: 공식 힌트의 명령 파싱·상대 좌표·로봇별 분류 방향을 구현합니다. 공개된 출제자 MLP와 같은 코드라고 주장하지 않습니다.
- Potato: 공식 유사도 표·탐색 아이디어에 오차 바닥값을 갖는 사후확률 갱신과 정보량 기반 선택을 추가했습니다.
- Double Agent Dilemma: 공식 공격 목표와 전처리를 따르며 margin 손실·성공 후 축소·실패 영텐서는 본 구현의 설계입니다.
- Ghost: 공식 문장 임베딩 힌트에 학습 기반 구분자 탐색과 전역 경계 선택을 조합했습니다. 위원회의 미세조정 모델을 재현한 것은 아닙니다.
- Field: 공식 분기 모델·dropout 힌트를 바탕으로 affine I, 4-bit dropout, 보수적인 I 분류 문턱을 적용했습니다.

## 실측에 사용한 고정 자료

공식 문제·채점기 코드 revision은 [`e1963801`](https://github.com/IOAI-official/IOAI-2026/tree/e1963801aa4cf33a5a03520b39fa6620a7894c54)입니다. 로컬 재현 채점식과 원본의 관계는 [채점 출처 기록](benchmarks/scoring_provenance.json)에 적었습니다. Dilemma와 Field는 공식 평가 함수를 직접 불러옵니다.

| 자료 | 고정 revision |
|---|---|
| [Find the Order](https://huggingface.co/datasets/IOAI-official/ioai-2026-find-the-order/tree/71a35b5a566665e242416eba45f018dc378663b8) | `71a35b5a566665e242416eba45f018dc378663b8` |
| [Robot Chasing](https://huggingface.co/datasets/IOAI-official/ioai-2026-robot-chasing/tree/8a82acbdfda387269649d4bcf580fbdab1ed3cc6) | `8a82acbdfda387269649d4bcf580fbdab1ed3cc6` |
| [Potato](https://huggingface.co/datasets/IOAI-official/ioai-2026-potato/tree/122716fc56a212e58b1756158bdeb4c2ea2da5e6) | `122716fc56a212e58b1756158bdeb4c2ea2da5e6` |
| [Double Agent Dilemma](https://huggingface.co/datasets/IOAI-official/ioai-2026-double-agent-dilemma/tree/6f42de8f02c801618d0b90189bf4c759a9cd6381) | `6f42de8f02c801618d0b90189bf4c759a9cd6381` |
| [Ghost of the Machine](https://huggingface.co/datasets/IOAI-official/ioai-2026-ghost-of-the-machine/tree/6dcd17d529f154935cda550c6e80f9801887a316) | `6dcd17d529f154935cda550c6e80f9801887a316` |
| [IOAI Field](https://huggingface.co/datasets/IOAI-official/ioai-2026-ioai-field/tree/35136ba0abb05ea8dcdb6ec3db200617a9da6104) | `35136ba0abb05ea8dcdb6ec3db200617a9da6104` |

Whisper, Qwen, BGE, ViT의 모델 revision과 다운로드 파일 SHA-256은 각 benchmark JSON·다운로드 manifest에 남겼습니다. 공개 저장소에 올라온 평가 자료만 사용하며, 데이터·정답·모델 가중치는 이 저장소에 포함하지 않습니다.

