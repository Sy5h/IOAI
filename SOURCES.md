# 공식 자료와 출처

확인 기준: 2026-10-15. 온라인 원본은 이후 갱신될 수 있습니다.

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

