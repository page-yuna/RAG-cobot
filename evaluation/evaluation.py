import pandas as pd

# 데이터 불러오기
file_path = "./cobot_chatbot_qa.csv"
df = pd.read_csv(file_path)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
import plotly.express as px

# BLEU, ROUGE, BERTScore 저장용 리스트
bleu_scores = []
rouge_l_scores = []
references = df["Ground-truth 정답 후보"].tolist()
candidates = df["챗봇 답변"].tolist()

# BLEU와 ROUGE 계산
smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

for ref, cand in zip(references, candidates):
    # BLEU
    bleu = sentence_bleu([ref.split()], cand.split(), smoothing_function=smoothie)
    bleu_scores.append(bleu)
    
    # ROUGE-L
    rouge_score = rouge.score(ref, cand)['rougeL'].fmeasure
    rouge_l_scores.append(rouge_score)

# BERTScore 계산
P, R, F1 = bert_score.score(candidates, references, lang='ko')

# 결과 정리
results_df = pd.DataFrame({
    "질문": df["질문"],
    "BLEU": bleu_scores,
    "ROUGE-L": rouge_l_scores,
    "BERTScore(F1)": F1.numpy()
})

# 평균 성능도 계산
bleu_avg = sum(bleu_scores) / len(bleu_scores)
rouge_avg = sum(rouge_l_scores) / len(rouge_l_scores)
bertscore_avg = F1.mean().item()

(bleu_avg, rouge_avg, bertscore_avg)
