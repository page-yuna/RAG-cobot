#rag_final.py
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# 1. 모델 설정 (4bit 양자화)
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    do_sample=True, #경고 무시하기 위해 true로 설정
    repetition_penalty=1.03,
    pad_token_id=tokenizer.eos_token_id,
)
llm = HuggingFacePipeline(pipeline=text_generator)

# 2. 데이터 불러오기
csv_path = "/webcrawling_data.csv"
df = pd.read_csv(csv_path, encoding="ISO-8859-1")

# 3. 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
combined_texts = (df["question"].fillna("") + " " + df["answer"].fillna("")).tolist()
docs = text_splitter.create_documents(combined_texts)

# 4. 벡터스토어
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 5. 프롬프트 템플릿
prompt_template = """
제공된 다음 문맥 정보를 바탕으로, 사용자 질문에 대한 **한국어로 된 간결한 답변**만 작성해주세요.
다른 정보나 서론, 추가 설명은 일절 포함하지 마세요. 답변은 오직 한국어로만 구성되어야 합니다.

문맥:
{context}

사용자 질문: {question}

답변:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 6. QA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    input_key="question",
    chain_type_kwargs={"prompt": PROMPT},
)

# 7. 질의 실행
if __name__ == "__main__":
    while True:
        user_query = input("\n❓ 질문 입력 (종료: exit): ")
        if user_query.strip().lower() == "exit":
            print("\n🔚 종료합니다.")
            break

        result = qa_chain.invoke({"question": user_query})
        response_text = result['result']

        if "답변:" in response_text:
            cleaned = response_text.split("답변:", 1)[1].strip()
        else:
            cleaned = response_text.strip()

        print("\n🟢 답변:", cleaned)
