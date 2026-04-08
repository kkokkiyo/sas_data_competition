import pandas as pd
import numpy as np

print("데이터 불러오는 중...")
df_fin = pd.read_csv('train/train_finance_profile.csv')
df_target = pd.read_csv('train/train_targets.csv')
df_trans = pd.read_csv('train/train_transaction_history.csv')

# 금융 데이터와 타겟(정답) 데이터 결합
df_explore = pd.merge(df_fin, df_target, on='customer_id')

print("\n=======================================================")
print("💡 1. 이탈(Churn) 고객 vs 유지 고객의 금융 자산 차이")
print("=======================================================")
# 이탈 여부(0, 1)에 따른 금융 변수들의 평균값 비교
churn_diff = df_explore.groupby('target_churn')[['credit_score', 'total_deposit_balance', 'card_loan_amt', 'fin_overdue_days']].mean().round(2)
print(churn_diff)
print("\n해석 포인트: 1(이탈)인 사람들의 card_loan_amt(카드론)나 fin_overdue_days(연체일수)가 유지 고객보다 훨씬 높지 않은가?")

print("\n=======================================================")
print("💡 2. 초우량 VIP 고객(LTV 상위 1%)은 무엇이 다를까?")
print("=======================================================")
# LTV 상위 1% 기준 금액 찾기
vip_threshold = df_explore['target_ltv'].quantile(0.99)
print(f"상위 1% VIP 커트라인: {vip_threshold:,.0f}원")

# VIP와 일반 고객 나누기
df_explore['is_vip'] = np.where(df_explore['target_ltv'] >= vip_threshold, 'VIP', 'Normal')

# VIP와 일반 고객의 차이 비교
vip_diff = df_explore.groupby('is_vip')[['credit_score', 'total_deposit_balance', 'num_active_cards']].mean().round(2)
print(vip_diff)
print("\n해석 포인트: VIP들은 예적금(total_deposit_balance)이 압도적으로 많거나, 신용카드 개수(num_active_cards)가 다르지 않은가?")

print("\n=======================================================")
print("💡 3. 구매 카테고리별 이탈률 확인")
print("=======================================================")
# 고객별로 '가장 많이 구매한 카테고리' 추출
top_category = df_trans.groupby(['customer_id', 'item_category'])['trans_id'].count().reset_index()
top_category = top_category.sort_values(['customer_id', 'trans_id'], ascending=[True, False]).drop_duplicates('customer_id')

cat_explore = pd.merge(top_category, df_target, on='customer_id')
cat_churn_rate = cat_explore.groupby('item_category')['target_churn'].mean().round(4) * 100
print("주력 카테고리별 이탈률(%):")
print(cat_churn_rate.sort_values(ascending=False))
print("\n해석 포인트: 특정 카테고리(예: Fashion)를 주로 사는 사람들의 이탈률이 유독 높다면, 모델에 '주력 카테고리' 변수를 만들어야 함!")