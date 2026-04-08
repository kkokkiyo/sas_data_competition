import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_squared_error

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
# 2. 데이터 불러오기 및 마스터 테이블(master_df) 생성
# ========================================================
print("데이터를 불러오고 마스터 테이블을 만드는 중...")

# 파일 경로 주의 (train 폴더 안의 파일들)
df_cust = pd.read_csv('train/train_customer_info.csv')
df_trans = pd.read_csv('train/train_transaction_history.csv')
df_fin = pd.read_csv('train/train_finance_profile.csv')
df_target = pd.read_csv('train/train_targets.csv')

# 구매 이력 요약
trans_agg = df_trans.groupby('customer_id').agg(
    txn_count=('trans_id', 'count'),
    txn_sum=('trans_amount', 'sum'),
    txn_mean=('trans_amount', 'mean')
).reset_index()

# 온라인 구매 비율 계산
df_trans['is_online'] = (df_trans['biz_type'] == 'Online').astype(int)
online_ratio = df_trans.groupby('customer_id')['is_online'].mean().reset_index()
online_ratio.rename(columns={'is_online': 'online_ratio'}, inplace=True)

# 병합하여 마스터 테이블 완성
trans_features = pd.merge(trans_agg, online_ratio, on='customer_id')
master_df = df_cust.merge(trans_features, on='customer_id', how='left') \
                   .merge(df_fin, on='customer_id', how='left') \
                   .merge(df_target, on='customer_id', how='left')

# 파생 변수 추가
master_df['loan_to_deposit'] = master_df['total_loan_balance'] / (master_df['total_deposit_balance'] + 1)

print(f"마스터 테이블 완성! 크기: {master_df.shape}")

# ========================================================
# 3. 모델링을 위한 데이터 준비
# ========================================================
print("\n모델링 데이터 준비 중...")

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
# 4. [모델 1] Churn (이탈) 예측 모델 학습
# ========================================================
print("\n🌳 [1/2] 이탈(Churn) 예측 모델 학습 중...")
clf_churn = lgb.LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05, verbose=-1)
clf_churn.fit(X_train, y_churn_train)

# 확률 예측
pred_churn_val = clf_churn.predict_proba(X_val)[:, 1]

# ========================================================
# 5. [모델 2] LTV (생애가치) 예측 모델 학습
# ========================================================
print("🌳 [2/2] LTV 예측 모델 학습 중...")
y_ltv_train_log = np.log1p(y_ltv_train)

reg_ltv = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, verbose=-1)
reg_ltv.fit(X_train, y_ltv_train_log)

# 예측 후 로그 복원
pred_ltv_val_log = reg_ltv.predict(X_val)
pred_ltv_val = np.expm1(pred_ltv_val_log)

# ========================================================
# 6. 최종 점수 확인
# ========================================================
calculate_my_score(y_churn_val, pred_churn_val, y_ltv_val, pred_ltv_val)

# ========================================================
# 7. Test 데이터 추론 및 최종 제출(Submission) 파일 생성
# ========================================================
print("\n🚀 Test 데이터 추론 및 제출 파일 생성 중...")

# 7-1. Test 데이터 불러오기 (test 폴더 경로에 주의하세요)
try:
    test_cust = pd.read_csv('test/test_customer_info.csv')
    test_trans = pd.read_csv('test/test_transaction_history.csv')
    test_fin = pd.read_csv('test/test_finance_profile.csv')
except FileNotFoundError:
    print("❌ 오류: test 폴더 안에 파일이 없습니다. 경로를 확인해주세요.")
    exit()

# 7-2. Test 데이터용 마스터 테이블 생성 (Train과 완벽히 동일하게 진행해야 함!)
print("Test 데이터 마스터 테이블 생성 중...")
test_trans_agg = test_trans.groupby('customer_id').agg(
    txn_count=('trans_id', 'count'),
    txn_sum=('trans_amount', 'sum'),
    txn_mean=('trans_amount', 'mean')
).reset_index()

test_trans['is_online'] = (test_trans['biz_type'] == 'Online').astype(int)
test_online_ratio = test_trans.groupby('customer_id')['is_online'].mean().reset_index()
test_online_ratio.rename(columns={'is_online': 'online_ratio'}, inplace=True)

test_trans_features = pd.merge(test_trans_agg, test_online_ratio, on='customer_id')
test_master = test_cust.merge(test_trans_features, on='customer_id', how='left') \
                       .merge(test_fin, on='customer_id', how='left')

# 파생 변수 추가
test_master['loan_to_deposit'] = test_master['total_loan_balance'] / (test_master['total_deposit_balance'] + 1)

# 7-3. 범주형 변수 처리 및 모델 입력 데이터 준비
for col in cat_cols:
    test_master[col] = test_master[col].astype('category')

# Test 데이터에는 원래 정답(target)이 없으므로 customer_id와 join_date만 버립니다.
X_test = test_master.drop(columns=['customer_id', 'join_date'])

# 7-4. 학습된 모델로 실전 예측!
print("Test 데이터 이탈 및 LTV 예측 중...")
final_pred_churn = clf_churn.predict_proba(X_test)[:, 1]

# LTV 예측 및 로그 복원
final_pred_ltv_log = reg_ltv.predict(X_test)
final_pred_ltv = np.expm1(final_pred_ltv_log)

# 대회 규칙(안내문) 💡: "예상 LTV 금액 (음수 불가)" 조건을 지키기 위해 0보다 작은 값은 0으로 처리
final_pred_ltv = np.clip(final_pred_ltv, 0, None)

# 7-5. 제출 양식(DataFrame) 만들기
submission = pd.DataFrame({
    'customer_id': test_master['customer_id'],
    'target_churn': final_pred_churn,
    'target_ltv': final_pred_ltv
})

# 7-6. CSV 파일로 저장 (팀명 부분을 실제 팀 이름으로 바꿔주세요!)
submission_filename = 'Fukuoka_submission_2주차.csv'
submission.to_csv(submission_filename, index=False, encoding='utf-8')

print(f"🎉 제출 파일이 성공적으로 생성되었습니다: {submission_filename}")
print("이제 이 파일을 이메일(kdis21competition@gmail.com)로 제출하시면 됩니다!")