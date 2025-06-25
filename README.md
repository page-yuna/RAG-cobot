# 🤖 협동로봇 문제 해결을 위한 검색증강생성(RAG) 기반 챗봇 서비스 연구

본 프로젝트는 협동로봇(cobot) 사용자가 현장에서 겪는 기술적 문제를 해결하기 위해, **검색증강생성(RAG: Retrieval-Augmented Generation)** 기법을 적용한 **LLM 기반 챗봇**을 개발한 것입니다. 기술 문서 및 사용자 Q&A 데이터로부터 지식을 추출하고, 대규모 언어 모델(LLM)을 통해 실시간 맞춤형 지원을 제공합니다.


## 📌 프로젝트 개요

- **프로젝트명:** 협동로봇 문제해결을 위한 검색증강생성 기반 챗봇 서비스 연구
- **진행 기간:** 2025년 3월 ~ 2025년 6월
- **지도 교수:** 김형중 교수님


## 🛠 주요 기술 및 라이브러리

- **LLM**: ['mistralai/Mistral-7B-Instruct'](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- **문서 임베딩**: 'intfloat/multilingual-e5-small'
- **벡터 DB**: FAISS
- **RAG 구성**: LangChain 기반 커스텀 체인 ('ConversationalRetrievalQA' 기반 변형)
- **배포**: Gradio (Colab 기반 프로토타이핑)
- **데이터 처리**: Python (pandas, regex, numpy 등)


## 📁 프로젝트 구조

```bash
├── data/                 # 크롤링된 협동로봇 Q&A, 기술 문서
├── embedding/            # 문서 임베딩 처리 및 FAISS 저장
├── rag_pipeline/         # RAG 파이프라인 관련 코드
│   ├── embed_docs.py
│   ├── rag_chain.py
│   └── utils.py
├── interface/
│   └── gradio_ui.py      # Gradio 기반 챗봇 UI
├── evaluation/
│   ├── bleu_rouge_eval.py
│   └── llm_judge_eval.py
└── README.md             # 프로젝트 설명
```

---

## 🧩 시스템 구성도

> 시스템은 다음과 같은 흐름으로 동작합니다:

1. 사용자 질문 입력
2. 벡터 DB에서 유사 문서 검색
3. 검색 결과와 함께 LLM에 프롬프트 전달
4. 문맥 기반 응답 생성
5. 실시간 답변 제공

---

## 📊 성능 평가

* **BLEU / ROUGE 점수**: 기존 검색 기반 도구 대비 높은 일관성과 정보 충실도 확보
* **LLM-as-a-Judge 평가**: GPT-4를 통해 비교 평가 시 챗봇 응답의 정확성과 구체성 측면에서 우수
* **사용자 시나리오 테스트**: 실제 협동로봇 문제(예: 펜스 설치 기준, 비상 정지 버튼 작동 오류 등)에 대해 신속한 답변 생성 확인

---

## 🚀 실행 방법

### 1. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 문서 임베딩 및 벡터 DB 구축

```bash
python embedding/embed_docs.py
```

### 3. RAG 파이프라인 실행

```bash
python rag_pipeline/rag_chain.py
```

### 4. Gradio UI 실행

```bash
python interface/gradio_ui.py
```

---

## ✨ 예시 질문

* "로봇 안전펜스 설치 기준이 뭐야?"
* "비상 정지 버튼이 작동하지 않을 때 어떻게 해야 돼?"
* "로봇의 TCP 설정 방법 알려줘"

---

## 🙌 기여자

* 박유나 (팀장, RAG 파이프라인 및 LLM 튜닝)
* 김보미 (데이터 수집 및 전처리)
* 김동희 (크롤링 및 벡터화 처리)

---

## 📄 라이선스

본 프로젝트는 **교육 목적**으로 개발된 것으로, 상업적 이용은 금지됩니다.

---

## 📬 문의

* Email: [yunapark@example.com](mailto:yunapark@example.com)
* GitHub: [github.com/yunapark](https://github.com/yunapark)

```

필요한 경우 `requirements.txt`, 실행 명령어, 성능 평가 결과 시각화 등을 포함하여 확장할 수 있어요. 추가하고 싶은 항목이 있다면 말씀해 주세요!
```
