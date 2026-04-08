import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore') # 성가신 경고 메시지 숨기기

# ========================================================
# 1. 자체 채점 함수
# ========================================================
def calculate_my_score(y_true_churn, y_pred_churn, y_true_ltv, y_pred_ltv):
    sub_auc = roc_auc_score(y_true_churn, y_pred_churn)
    rmse = np.sqrt(mean_squared_error(y_true_ltv, y_pred_ltv))
    rmse_safe = max(rmse, 1e-10) 
    total_score = (0.5 * sub_auc) + (0.5 * (1 / (1 + np.log10(rmse_safe))))
    
    print("\n====== 📊 검증 데이터(Validation) 평가 지표 결과 ======")
    print(f"✅ Churn AUC   : {sub_auc:.5f}")
    print(f"✅ LTV RMSE    : {rmse:.5f}")
    print(f"🏆 Total Score : {total_score:.5f}")
    print("=========================================================")
    return total_score

# ========================================================
# 2. [Train] 데이터 불러오기 및 마스터 테이블(master_df) 생성
# ========================================================
print("🚀 [Train] 데이터 불러오고 마스터 테이블 만드는 중 (하이브리드 고도화)...")

# 파일 경로 주의 (train 폴더 안의 파일들)
df_cust = pd.read_csv('train/train_customer_info.csv')
df_trans = pd.read_csv('train/train_transaction_history.csv')
df_fin = pd.read_csv('train/train_finance_profile.csv')
df_target = pd.read_csv('train/train_targets.csv')

base_date = pd.to_datetime('2023-12-31')

# 💡 [파생 변수] 가입 기간(Tenure)
df_cust['join_date_dt'] = pd.to_datetime(df_cust['join_date'])
df_cust['tenure_days'] = (base_date - df_cust['join_date_dt']).dt.days
df_cust.drop(columns=['join_date_dt'], inplace=True) # 임시 날짜 컬럼 삭제

# 💡 [파생 변수] 최근성(Recency)
df_trans['trans_date'] = pd.to_datetime(df_trans['trans_date'])
last_purchase = df_trans.groupby('customer_id')['trans_date'].max().reset_index()
last_purchase['recency_days'] = (base_date - last_purchase['trans_date']).dt.days

# 구매 이력 요약 (기본)
trans_agg = df_trans.groupby('customer_id').agg(
    txn_count=('trans_id', 'count'),
    txn_sum=('trans_amount', 'sum'),
    txn_mean=('trans_amount', 'mean')
).reset_index()

# 💡 [파생 변수] 온라인 구매 비율 & 식료품(Grocery) 비율
df_trans['is_online'] = (df_trans['biz_type'] == 'Online').astype(int)
online_ratio = df_trans.groupby('customer_id')['is_online'].mean().reset_index()
online_ratio.rename(columns={'is_online': 'online_ratio'}, inplace=True)

df_trans['is_grocery'] = (df_trans['item_category'] == 'Grocery').astype(int)
grocery_ratio = df_trans.groupby('customer_id')['is_grocery'].mean().reset_index()
grocery_ratio.rename(columns={'is_grocery': 'grocery_ratio'}, inplace=True)

# 병합하여 마스터 테이블 완성
trans_features = pd.merge(trans_agg, online_ratio, on='customer_id')
trans_features = pd.merge(trans_features, grocery_ratio, on='customer_id', how='left')
trans_features = pd.merge(trans_features, last_purchase[['customer_id', 'recency_days']], on='customer_id', how='left')

master_df = df_cust.merge(trans_features, on='customer_id', how='left') \
                   .merge(df_fin, on='customer_id', how='left') \
                   .merge(df_target, on='customer_id', how='left')

# 💡 [파생 변수] 금융 건전성 및 적신호 변수들
master_df['loan_to_deposit'] = master_df['total_loan_balance'] / (master_df['total_deposit_balance'] + 1)
master_df['has_card_loan'] = (master_df['card_loan_amt'] > 0).astype(int)
master_df['is_overdue'] = (master_df['fin_overdue_days'] > 0).astype(int)
master_df['used_cash_service'] = (master_df['card_cash_service_amt'] > 0).astype(int)

# ========================================================
# 3. 모델링을 위한 데이터 준비
# ========================================================
drop_cols = ['customer_id', 'join_date', 'target_churn', 'target_ltv']
cat_cols = ['gender', 'region_code', 'prefer_category', 'income_group']

# 범주형 변수 변환
for col in cat_cols:
    master_df[col] = master_df[col].astype('category')

X = master_df.drop(columns=drop_cols)
y_churn = master_df['target_churn']
y_ltv = master_df['target_ltv']

# Train / Validation 분리 (8:2)
X_train, X_val, y_churn_train, y_churn_val, y_ltv_train, y_ltv_val = train_test_split(
    X, y_churn, y_ltv, test_size=0.2, random_state=42
)

# ========================================================
# 4. [모델 1] Churn 예측 (하이브리드 전략 1: 가중치 배제)
# ========================================================
print("\n🌳 [1/2] 이탈(Churn) 예측 모델 학습 중 (순수 분류력 집중)...")
# 💡 고의로 scale_pos_weight를 빼서 AUC 점수를 극대화합니다!
clf_churn = lgb.LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05, verbose=-1)
clf_churn.fit(X_train, y_churn_train)

pred_churn_val = clf_churn.predict_proba(X_val)[:, 1]

# ========================================================
# 5. [모델 2] LTV 예측 (하이브리드 전략 2: 체이닝 & 상호작용)
# ========================================================
print("🌳 [2/2] LTV 예측 모델 학습 중 (이탈 확률 체이닝 결합)...")
y_ltv_train_log = np.log1p(y_ltv_train)

# 학습 데이터의 Churn 확률 추출
train_churn_prob = clf_churn.predict_proba(X_train)[:, 1]

# LTV 예측용 Feature 복사본 생성
X_train_ltv = X_train.copy()
X_val_ltv = X_val.copy()

# 💡 모델 체이닝 (확률 추가)
X_train_ltv['pred_churn_prob'] = train_churn_prob
X_val_ltv['pred_churn_prob'] = pred_churn_val 

# 💡 상호작용 변수: "이탈 확률을 고려한 기대 매출"
X_train_ltv['expected_survival_value'] = X_train_ltv['txn_sum'] * (1 - X_train_ltv['pred_churn_prob'])
X_val_ltv['expected_survival_value'] = X_val_ltv['txn_sum'] * (1 - X_val_ltv['pred_churn_prob'])

# 💡 이탈 위험군 계급화 변수
X_train_ltv['churn_risk_group'] = pd.cut(X_train_ltv['pred_churn_prob'], bins=[-1, 0.2, 0.5, 1.1], labels=[0, 1, 2]).astype(int)
X_val_ltv['churn_risk_group'] = pd.cut(X_val_ltv['pred_churn_prob'], bins=[-1, 0.2, 0.5, 1.1], labels=[0, 1, 2]).astype(int)

# 학습 및 예측
reg_ltv = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, verbose=-1)
reg_ltv.fit(X_train_ltv, y_ltv_train_log)

pred_ltv_val_log = reg_ltv.predict(X_val_ltv)
pred_ltv_val = np.expm1(pred_ltv_val_log)

# ========================================================
# 6. 최종 점수 확인
# ========================================================
calculate_my_score(y_churn_val, pred_churn_val, y_ltv_val, pred_ltv_val)

# ========================================================
# 7. [Test] 실전 데이터 추론 및 최종 제출 파일 생성
# ========================================================
print("\n🚀 [Test] 실전 데이터 추론 및 제출 파일 생성 중...")

try:
    test_cust = pd.read_csv('test/test_customer_info.csv')
    test_trans = pd.read_csv('test/test_transaction_history.csv')
    test_fin = pd.read_csv('test/test_finance_profile.csv')
except FileNotFoundError:
    print("❌ 오류: test 폴더 안에 파일이 없습니다. 경로를 확인해주세요.")
    exit()

# --- Test 데이터도 Train과 완벽히 동일하게 파생 변수 생성 ---
test_cust['join_date_dt'] = pd.to_datetime(test_cust['join_date'])
test_cust['tenure_days'] = (base_date - test_cust['join_date_dt']).dt.days
test_cust.drop(columns=['join_date_dt'], inplace=True)

test_trans['trans_date'] = pd.to_datetime(test_trans['trans_date'])
test_last_purchase = test_trans.groupby('customer_id')['trans_date'].max().reset_index()
test_last_purchase['recency_days'] = (base_date - test_last_purchase['trans_date']).dt.days

test_trans_agg = test_trans.groupby('customer_id').agg(
    txn_count=('trans_id', 'count'),
    txn_sum=('trans_amount', 'sum'),
    txn_mean=('trans_amount', 'mean')
).reset_index()

test_trans['is_online'] = (test_trans['biz_type'] == 'Online').astype(int)
test_online_ratio = test_trans.groupby('customer_id')['is_online'].mean().reset_index()
test_online_ratio.rename(columns={'is_online': 'online_ratio'}, inplace=True)

test_trans['is_grocery'] = (test_trans['item_category'] == 'Grocery').astype(int)
test_grocery_ratio = test_trans.groupby('customer_id')['is_grocery'].mean().reset_index()
test_grocery_ratio.rename(columns={'is_grocery': 'grocery_ratio'}, inplace=True)

test_trans_features = pd.merge(test_trans_agg, test_online_ratio, on='customer_id')
test_trans_features = pd.merge(test_trans_features, test_grocery_ratio, on='customer_id', how='left')
test_trans_features = pd.merge(test_trans_features, test_last_purchase[['customer_id', 'recency_days']], on='customer_id', how='left')

test_master = test_cust.merge(test_trans_features, on='customer_id', how='left') \
                       .merge(test_fin, on='customer_id', how='left')

test_master['loan_to_deposit'] = test_master['total_loan_balance'] / (test_master['total_deposit_balance'] + 1)
test_master['has_card_loan'] = (test_master['card_loan_amt'] > 0).astype(int)
test_master['is_overdue'] = (test_master['fin_overdue_days'] > 0).astype(int)
test_master['used_cash_service'] = (test_master['card_cash_service_amt'] > 0).astype(int)

# 범주형 처리
for col in cat_cols:
    test_master[col] = test_master[col].astype('category')

X_test = test_master.drop(columns=['customer_id', 'join_date'])

# --- 💡 Test 실전 추론 (하이브리드 전략 적용) ---
# 1) Churn 예측
final_pred_churn = clf_churn.predict_proba(X_test)[:, 1]

# 2) LTV 모델을 위한 Feature 세팅
X_test_ltv = X_test.copy()
X_test_ltv['pred_churn_prob'] = final_pred_churn
X_test_ltv['expected_survival_value'] = X_test_ltv['txn_sum'] * (1 - X_test_ltv['pred_churn_prob'])
X_test_ltv['churn_risk_group'] = pd.cut(X_test_ltv['pred_churn_prob'], bins=[-1, 0.2, 0.5, 1.1], labels=[0, 1, 2]).astype(int)

# 3) LTV 예측
final_pred_ltv_log = reg_ltv.predict(X_test_ltv)
final_pred_ltv = np.expm1(final_pred_ltv_log)
final_pred_ltv = np.clip(final_pred_ltv, 0, None) # 음수 방지

# 7-5. 제출 양식(DataFrame) 만들기
submission = pd.DataFrame({
    'customer_id': test_master['customer_id'],
    'target_churn': final_pred_churn,
    'target_ltv': final_pred_ltv
})

# 7-6. CSV 파일로 저장
submission_filename = 'Fukuoka_submission_최종하이브리드.csv'
submission.to_csv(submission_filename, index=False, encoding='utf-8')

print(f"🎉 성공적으로 생성 완료! 제출 파일명: {submission_filename}")
print("수고하셨습니다. 역대급 점수를 기대하셔도 좋습니다!")