# %% [1] 라이브러리 로드 및 데이터 읽기
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

print("데이터 로딩 중...")

# Train 데이터 (앞에 'train/' 폴더 경로 추가)
tr_info = pd.read_csv('train/train_customer_info.csv')
tr_fin = pd.read_csv('train/train_finance_profile.csv')
tr_trans = pd.read_csv('train/train_transaction_history.csv')
tr_target = pd.read_csv('train/train_targets.csv')

# Test 데이터 (앞에 'test/' 폴더 경로 추가)
te_info = pd.read_csv('test/test_customer_info.csv')
te_fin = pd.read_csv('test/test_finance_profile.csv')
te_trans = pd.read_csv('test/test_transaction_history.csv')

# %% [2] 거래 내역(Transaction) 요약 전처리 (Feature Engineering)
print("거래 내역 요약 중...")

def agg_transaction(df):
    """시계열 거래 내역을 고객별 통계로 압축하는 함수"""
    agg_df = df.groupby('customer_id').agg(
        total_trans_cnt=('trans_id', 'count'),          # 총 거래 횟수
        total_trans_amt=('trans_amount', 'sum'),        # 총 거래 금액
        mean_trans_amt=('trans_amount', 'mean'),        # 평균 거래 금액
        online_trans_cnt=('biz_type', lambda x: (x == 'Online').sum()) # 온라인 결제 횟수
    ).reset_index()
    return agg_df

tr_trans_agg = agg_transaction(tr_trans)
te_trans_agg = agg_transaction(te_trans)

# %% [3] 데이터 병합 (Merge)
print("데이터 테이블 병합 중...")

def merge_data(info, fin, trans_agg, target=None):
    """고객정보 + 금융 + 요약된 거래내역(+정답)을 합치는 함수"""
    df = pd.merge(info, fin, on='customer_id', how='left')
    df = pd.merge(df, trans_agg, on='customer_id', how='left')
    
    if target is not None:
        df = pd.merge(df, target, on='customer_id', how='left')
    return df

train = merge_data(tr_info, tr_fin, tr_trans_agg, tr_target)
test = merge_data(te_info, te_fin, te_trans_agg)

# 범주형 변수(문자열) Label Encoding
# 수정 후 ('income_group' 추가)
cat_cols = ['gender', 'region_code', 'prefer_category', 'income_group']
for col in cat_cols:
    le = LabelEncoder()
    # train과 test를 합쳐서 핏팅하여 에러 방지
    le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 날짜(join_date) 처리: 단순 베이스라인이므로 일단 제외
train = train.drop('join_date', axis=1)
test_id = test['customer_id'] # 최종 제출용 ID 저장
test = test.drop(['customer_id', 'join_date'], axis=1)

# 결측치 0으로 채우기
train = train.fillna(0)
test = test.fillna(0)

print(f"병합 완료! Train 형태: {train.shape}, Test 형태: {test.shape}")

# %% [4] 투트랙(Two-Track) 모델링 및 학습
print("\n[Track 1] 이탈(Churn) 예측 분류 모델 학습 중...")
X = train.drop(['customer_id', 'target_churn', 'target_ltv'], axis=1)
y_churn = train['target_churn']
y_ltv = train['target_ltv']

# 검증 세트 분리
X_tr, X_val, y_churn_tr, y_churn_val, y_ltv_tr, y_ltv_val = train_test_split(
    X, y_churn, y_ltv, test_size=0.2, random_state=42
)

# 모델 1: 이탈 예측 (분류)
model_churn = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model_churn.fit(X_tr, y_churn_tr)
churn_val_pred = model_churn.predict_proba(X_val)[:, 1] # 확률값 추출
print(f"Churn 검증 AUC: {roc_auc_score(y_churn_val, churn_val_pred):.4f}")

print("\n[Track 2] LTV 예측 회귀 모델 학습 중...")
# 모델 2: LTV 예측 (회귀)
model_ltv = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_ltv.fit(X_tr, y_ltv_tr)
ltv_val_pred = model_ltv.predict(X_val)
print(f"LTV 검증 RMSE: {np.sqrt(mean_squared_error(y_ltv_val, ltv_val_pred)):.4f}")

# %% [5] 최종 예측 및 제출 파일 생성
print("\n테스트 데이터 예측 및 제출 파일 생성 중...")
# Test 데이터 예측
test_churn_pred = model_churn.predict_proba(test)[:, 1]
test_ltv_pred = model_ltv.predict(test)

# 음수 LTV 예측값 방어 (제출 규정)
test_ltv_pred = np.maximum(test_ltv_pred, 0)

# 양식에 맞춰 제출 파일 생성
submission = pd.DataFrame({
    'customer_id': test_id,
    'target_churn': test_churn_pred,
    'target_ltv': test_ltv_pred
})

submission.to_csv('advanced_submission_1.csv', index=False)
print("완료! 'advanced_submission_1.csv' 파일이 생성되었습니다.")