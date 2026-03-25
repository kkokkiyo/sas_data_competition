import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor # 강력한 부스팅 모델인 XGBoost 사용
from sklearn.metrics import mean_squared_error

# 1. 데이터 로드
train = pd.read_csv('TRAIN_DATA.csv', encoding='cp949')
test = pd.read_csv('TEST_DATA.csv', encoding='cp949')

TARGET_COL = 'TOTAL_ELEC' # 실제 타겟 변수명 유지

# ==========================================
# 💡 [NEW] 파생 변수 생성 (Feature Engineering)
# ==========================================
# 'FAC_'로 시작하는 모든 시설물 컬럼의 값을 더해 'TOTAL_FAC'라는 새로운 열을 만듭니다.
fac_cols = [col for col in train.columns if col.startswith('FAC_')]
train['TOTAL_FAC'] = train[fac_cols].sum(axis=1)
test['TOTAL_FAC'] = test[fac_cols].sum(axis=1)

# 건물당 평균 가스 사용량 (건물 수가 0일 경우를 대비해 약간의 숫자를 더해 나눔)
if 'TOTAL_GAS' in train.columns and TARGET_COL != 'TOTAL_GAS':
    train['GAS_PER_BLDG'] = train['TOTAL_GAS'] / (train['TOTAL_BIDG'] + 1e-5)
    test['GAS_PER_BLDG'] = test['TOTAL_GAS'] / (test['TOTAL_BIDG'] + 1e-5)

# ==========================================
# 3. 데이터 전처리 (결측치 및 문자열 변수 처리)
# ==========================================
categorical_cols = train.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if col == TARGET_COL:
        continue
    
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
    
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

train = train.fillna(0) 
test = test.fillna(0)

# 4. 데이터 분리
X = train.drop(TARGET_COL, axis=1)
y = train[TARGET_COL]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 💡 [NEW] XGBoost 모델 정의 및 하이퍼파라미터 튜닝
# ==========================================
# n_estimators: 나무의 개수, learning_rate: 학습률, max_depth: 나무의 깊이
model = XGBRegressor(
    n_estimators=500,       
    learning_rate=0.05,     
    max_depth=6,            
    random_state=42,
    tree_method='hist', 
    early_stopping_rounds=50 # <--- fit()에 있던 것을 이쪽으로 옮겼습니다!
)

# 조기 종료(Early Stopping) 적용: 검증 세트 성능이 50번 동안 안 오르면 학습 중단
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50 # 50번마다 학습 과정 출력
)

# 6. 자체 검증 성능 확인 (RMSE)
val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f'\n🔥 고도화 모델 검증 세트 RMSE: {rmse:.4f}')

# 7. 최종 예측 및 제출 파일 생성
if TARGET_COL in test.columns:
    test = test.drop(TARGET_COL, axis=1)
    
test_pred = model.predict(test)

submission = pd.DataFrame({
    TARGET_COL: test_pred
})

submission.to_csv('advanced_submission.csv', index=False, encoding='cp949')
print("제출 파일 생성 완료: advanced_submission.csv")