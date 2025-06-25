from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 질문 번호
q_ids = [f"Q{i+1}" for i in range(len(results_df))]
results_df["Total"] = results_df[["BLEU", "ROUGE-L", "BERTScore(F1)"]].sum(axis=1)

# subplot 생성
fig = make_subplots(
    rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.15,
    subplot_titles=("BLEU", "ROUGE-L", "BERTScore(F1)", "총합 점수 (Total)")
)

# 각 plot에 서로 다른 색상 적용
fig.add_trace(go.Bar(x=q_ids, y=results_df["BLEU"], name="BLEU", marker_color='royalblue'), row=1, col=1)
fig.add_trace(go.Bar(x=q_ids, y=results_df["ROUGE-L"], name="ROUGE-L", marker_color='royalblue'), row=1, col=2)
fig.add_trace(go.Bar(x=q_ids, y=results_df["BERTScore(F1)"], name="BERTScore(F1)", marker_color='royalblue'), row=2, col=1)
fig.add_trace(go.Bar(x=q_ids, y=results_df["Total"], name="Total", marker_color='royalblue'), row=2, col=2)

# Y축 범위 자동 또는 필요 시 range 설정 가능
fig.update_yaxes(title_text="점수", row=1, col=1)
fig.update_yaxes(title_text="점수", row=1, col=2)
fig.update_yaxes(title_text="점수", row=2, col=1)
fig.update_yaxes(title_text="점수", row=2, col=2)

# 공통 레이아웃
fig.update_layout(
    height=800,
    width=1000,
    title_text="챗봇 응답 성능 비교 (BLEU / ROUGE-L / BERTScore / 총합)",
    showlegend=False
)

fig.write_html("rag_answer_scores.html")
