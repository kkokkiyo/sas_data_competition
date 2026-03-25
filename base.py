import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor # 회귀 모델로 변경
from sklearn.metrics import mean_squared_error # 회귀 평가 지표(MSE) 사용

# 1. 데이터 로드 (한글 깨짐 방지 인코딩 적용)
train = pd.read_csv('TRAIN_DATA.csv', encoding='cp949')
test = pd.read_csv('TEST_DATA.csv', encoding='cp949')

# 2. 타겟 변수 설정
# '데이터 소개.pdf'를 확인하시고 실제 예측해야 하는 컬럼명으로 꼭 변경해 주세요!
# (예: TOTAL_GAS, CMRC_GAS, TOTAL_ELEC 중 하나)
TARGET_COL = 'TOTAL_ELEC' 

# 3. 데이터 전처리 (결측치 및 문자열 변수 처리)
# AREA_NM(지역명), DIST_NM(행정동명) 같은 문자열 변수를 숫자로 변환 (Label Encoding)
categorical_cols = train.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if col == TARGET_COL:
        continue
    
    le = LabelEncoder()
    # train과 test에 있는 모든 문자열 종류를 합쳐서 학습
    le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
    
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 결측치(NaN) 처리 - 간단히 0으로 채웁니다.
train = train.fillna(0) 
test = test.fillna(0)

# 4. 모델 학습을 위한 데이터 분리
X = train.drop(TARGET_COL, axis=1)
y = train[TARGET_COL]

# 검증 세트 분리 (8:2 비율)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 회귀 모델 정의 및 학습
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 6. 자체 검증 성능 확인 (RMSE - 평균 제곱근 오차)
val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f'검증 세트 RMSE (오차 범위): {rmse:.4f}')

# 7. 최종 예측 및 제출 파일 생성
# 만약 테스트 데이터에도 타겟 컬럼이 빈칸으로 존재한다면 예측을 위해 제거해 줍니다.
if TARGET_COL in test.columns:
    test = test.drop(TARGET_COL, axis=1)
    
test_pred = model.predict(test)

# 제출 양식 데이터프레임 생성
submission = pd.DataFrame({
    TARGET_COL: test_pred
})

# 결과를 csv로 저장
submission.to_csv('baseline_submission.csv', index=False, encoding='cp949')
print("제출 파일 생성 완료: baseline_submission.csv")