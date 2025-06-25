import pandas as pd
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu

# ----------------------
# 평가 함수
# ----------------------
def evaluate_response(predicted, reference):
    ref_tokens = [reference.split()]
    pred_tokens = predicted.split()
    try:
        bleu = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5))
    except ZeroDivisionError:
        bleu = 0.0

    P, R, F1 = score([predicted], [reference], lang='en', verbose=False)
    bert_score_f1 = F1.item()

    return round(bleu, 4), round(bert_score_f1, 4)

# ----------------------
# 메인 처리 함수
# ----------------------
def filter_responses(file_path, output_path, min_bert_score=0.65):
    df = pd.read_excel(file_path)
    df = df[['Category', 'Cleaned Question', 'Cleaned Answer']].dropna()
    df = df.rename(columns={
        'Cleaned Question': 'question',
        'Cleaned Answer': 'answer'
    })

    bleu_list = []
    bert_list = []

    for _, row in df.iterrows():
        bleu, bert = evaluate_response(row['answer'], row['question'])
        bleu_list.append(bleu)
        bert_list.append(bert)

    df['BLEU'] = bleu_list
    df['BERT'] = bert_list

    # 필터링
    filtered_df = df[df['BERT'] >= min_bert_score].reset_index(drop=True)

    # 번호 열 추가
    filtered_df.index.name = 'No.'
    filtered_df.reset_index(inplace=True)

    # 엑셀 저장
    filtered_df.to_excel(output_path, index=False)
    print(f"✅ 필터링 완료: {len(filtered_df)}개 유지됨. 저장됨 → {output_path}")

# ----------------------
# 실행 파트
# ----------------------
if __name__ == "__main__":
    input_file = r"기존 파일위치"
    output_file = r"필터링 한 파일위치"
    filter_responses(input_file, output_file)
