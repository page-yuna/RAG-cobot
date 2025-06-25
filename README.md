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
├── data.csv                   # 협동로봇 문제 관련 Q&A 데이터
├── rag_final.py               # RAG 파이프라인 관련 코드
├── ragchat_output.csv         # RAG 기반 챗봇 실행 결과 파일
├── evaluation/                # RAG 응답 평가 관련 코드
│   ├── eval.py
│   └── eval_visualization.py
└── README.md                  # 프로젝트 설명
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


## ✨ 예시 질문

* "Polyscope 5.12에서 외부 TCP 보정을 사용할 수 있어?"
* "비상 정지 버튼이 작동하지 않을 때 어떻게 해야 돼?"
* "URCap 개발 시 C207A0 오류는 무엇을 의미하나요?"

---

## 📬 문의

* Email: [lxnx.llxnx@gmail.com](mailto:lxnx.llxnx@gmail.com)
* GitHub: [github.com/page-yuna](https://github.com/page-yuna)
