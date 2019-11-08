# Book-ML-PerfectGuide
Study : Book-ML-PerfectGuide

# 파이썬 머신러닝 완벽 가이드
## 01 파이썬 기반의 머신러닝과 생태계 이해
### 1. 머신러닝의 개념
#### 머신러닝의 분류
#### 데이터 전쟁
#### 파이썬과 R 기반의 머신러닝 비교
### 2. 파이썬 머신러닝 생태계를 구성하는 주요 패키지
#### 파이썬 머신러닝을 위한 S/W 설치
### 3. 넘파이
#### 넘파이 ndarray 개요
#### Ndarray의 데이터 타입
#### Ndarray를 편리하게 생성하기 - arange, zeros, ones
#### ndarray 의 차원과 크기를 변경하는 reshape()
#### 넘파이의 ndarray의 데이터 세트 선택하기 -인덱싱(indexing)
#### 행렬의 정렬 - sort()와 argsort()
#### 선형대수 연산 - 행렬 내적과 전치 행렬 구하기
### 4. 데이터 핸들링 - 판다스
#### 판다스 시작 - 파일을 DataFrame으로 로딩, 기본 API
#### DataFrame과 리스트, 딕셔너리, 넘파이 ndarray 상호 변환
#### DataFrame의 칼럼 데이터 세트 생성과 수정
#### DataFrame 데이터 삭제
#### Index 객체
#### 데이터 셀렉션 및 필터링
1. [] : 컬럼 기반 필터링 또는 불린 인덱싱 필터링 제공
2. ix[], loc[], iloc[]: 명칭/위치 기반 인덱싱을 제공
3. Boolean Indexing: 조건식에 따른 필터링을 제공
#### 정렬, Aggregation 함수, GroupBy 적용
1. Aggregation
- sum(), max(), min(), count()
- axis 0 and axis 1
2. groupby()
- DataFramGroupBy 객체
- df.groupby('Pclass')['Age'].agg([max, min])
- df.groupby('Pclass').agg({'Age':'max', 'Fare':'mean'})

#### 결손 데이터 처리하기
1. isna()
- df.isna().sum()
2. fillna()
- df['Cabin'].fillna('A')
- df['Embarked'].fillna(df['Embarked'].mean())
#### Apply lambda 식으로 데이터 가공
1. python lambda 
- lambda_squre = lambda x:x**2
- lambda_squre(3) -> 9
2. pandas apply lambda
- df['Name_len'] = df['Name'].apply(lambda x : len(x))

### 5. 정리

## 02 사이킷런으로 시작하는 머신러닝

### 1. 사이킷런 소개와 특징
1. numpy, scipy, ...
### 2. 첫 번째 머신러닝 만들어 보기 - 붓꽃 품종 예측하기
1. train test target prediction accuracy
### 3. 사이킷런의 기반 프레임워크 익히기
#### Estimator 이해 및 fit(), predict() 메서드
#### 사이킷런의 주요 모듈
#### 내장된 예제 데이터 세트
### 4. Model Selection 모듈 소개
#### 학습/테스트 데이터 세트 분리 - train_test_split()
#### 교차 검증
1. 학습 데이터 세트를 학습 데이터 세트와 검증 데이터 세트로 분할
2. K-Fold 교차 검증
- K개의 폴드 세트에 K번의 학습과 검증 평가 반복 수행
- 일반 K폴드 / Stratified K폴드
- Stratified K폴드: 불균형한 분포도를 가진 레이블(결정 클래스) 데이터 집합을 위한 K폴드 방식. 학습 데이터와 검증 데이터 세트가 가지는 레이블 분포도가 유사하도록 검증 데이터 추출
#### GridSearchCV - 교차 검증과 최적 하이퍼 파라미터 튜닝을 한 번에
### 5. 데이터 전처리
#### 데이터 인코딩
1. 원-핫 인코딩 pd.get_dummies(DatFrame)
#### 피처 스케일링과 정규화
#### StandardScalar
- 평균이 0이고 분산이 1인 가우시안 정규 분포를 가진 값으로 변환
- 정규화는 서로 다른 피처의 크기를 통일하기 위해 크기를 변환해 주는 개념
#### MinMaxScaler
- 데이터값을 0과 1사이의 범위 값으로 변환(음수 값이 있으면 -1에서 1값으로 변환)
### 6. 사이킷런으로 수행하는 타이타닉 생존자 예측
1. 데이터 전처리
- Null 처리
- 불필요 속성 제거
- 인코딩 수행
2. 모델 학습 및 검증/예측/평가
- 결정트리, 랜덤포레스트, 로지스틱회귀 학습 비교
- k폴드 교차 검증
- cross_val_score()와 GridSearchCV() 수행
### 7. 정리
- 데이터 전처리 > 데이터 세트 분리 > 모델 학습 및 검증 평가 > 예측 수행 > 평가

## 03 평가
### 1. 정확도(Accuracy)
### 2. 오차 행렬
- Confusion Matrix
- 예측 Negative/Positive 와 실제 Negative/Positive 구분
- TN(TrueNegative), TP(TruePositive), FN(FalseNegative), FP(FalsePositive)

### 3. 정밀도와 재현율
- Precision, Recall
- TP/FP+TP, TP/FN+TP
#### 정밀도/재현율 트레이드오프
#### 정밀도와 재현율의 맹점
### 4. F1 스코어
### 5. ROC 곡선과 AUC
### 6. 피마 인디언 당뇨병 예측
### 7. 정리
<테스트>
## 04 분류
### 1. 분류(Classification)의 개요
### 2. 결정 트리
#### 결정 트리 모델의 특징
#### 결정 트리 파라미터
#### 결정 트리 모델의 시각화
#### 결정 트리 과적합(Overfitting)
#### 결정 트리 실습 - 사용자 행동 인식 데이터 세트
### 3. 앙상블 학습
#### 앙상블 학습 개요
#### 보팅 유형 - 하드 보팅(Hard Voting)과 소프트 보팅(Soft Voting)
#### 보팅 분류기(Voting Classifier)
### 4. 랜덤 포레스트
#### 랜덤 포레스트의 개요 및 실습
#### 랜덤 포레스트 하이퍼 파라미터 및 튜닝
### 5. GBM(Gradient Boosting Machine)
#### GBM의 개요 및 실습
#### GBM 하이퍼 파라미터 및 튜닝
### 6. XGBoost(eXtra Gradient Boost)
#### XGBoost 개요
#### XGBoost 설치하기
#### 파이썬 래퍼 XGBoost 하이퍼 파라미터
#### 파이썬 래퍼 XGBoost 적용 - 위스콘신 유방암 예측
#### 사이킷런 래퍼 XGBoost의 개요 및 적용
### 7. LightGBM
#### LightGBM 설치
#### LightGBM 하이퍼 파라미터
#### 하이퍼 파라미터 튜닝 방안
#### 파이썬 래퍼 LightGBM과 사이킷런 래퍼 XGBoost,
#### LightGBM 하이퍼 파라미터 비교
#### LightGBM 적용 - 위스콘신 유방암 예측
### 8. 분류 실습 - 캐글 산탄데르 고객 만족 예측
#### 데이터 전처리
#### XGBoost 모델 학습과 하이퍼 파라미터 튜닝
#### LightGBM 모델 학습과 하이퍼 파라미터 튜닝
### 9. 분류 실습 - 캐글 신용카드 사기 검출
#### 언더 샘플링과 오버 샘플링의 이해
#### 데이터 일차 가공 및 모델 학습/예측/평가
#### 데이터 분포도 변환 후 모델 학습/예측/평가
#### 이상치 데이터 제거 후 모델 학습/예측/평가
#### SMOTE 오버 샘플링 적용 후 모델 학습/예측/평가
### 10. 스태킹 앙상블
#### 기본 스태킹 모델
#### CV 세트 기반의 스태킹
### 11. 정리
## 05 회귀
### 1. 회귀 소개
### 2. 단순 선형 회귀를 통한 회귀 이해
### 3. 비용 최소화하기 - 경사 하강법(Gradient Descent) 소개
### 4. 사이킷런 LinearRegression을 이용한 보스턴 주택 가격 예측
#### LinearRegression 클래스 - Ordinary Least Squares
#### 회귀 평가 지표
#### LinearRegression을 이용해 보스턴 주택 가격 회귀 구현
### 5. 다항 회귀와 과(대)적합/과소적합 이해
#### 다항 회귀 이해
#### 다항 회귀를 이용한 과소적합 및 과적합 이해
#### 편향-분산 트레이드오프(Bias-Variance Trade off)
### 6. 규제 선형 모델 - 릿지, 라쏘, 엘라스틱넷
#### 규제 선형 모델의 개요
#### 릿지 회귀
#### 라쏘 회귀
#### 엘라스틱넷 회귀
#### 선형 회귀 모델을 위한 데이터 변환
### 7. 로지스틱 회귀
### 8. 회귀 트리
### 9. 회귀 실습 - 자전거 대여 수요 예측
#### 데이터 클렌징 및 가공
#### 로그 변환, 피처 인코딩과 모델 학습/예측/평가
### 10. 회귀 실습 - 캐글 주택 가격: 고급 회귀 기법
#### 데이터 사전 처리(Preprocessing)
#### 선형 회귀 모델 학습/예측/평가
#### 회귀 트리 모델 학습/예측/평가
#### 회귀 모델의 예측 결과 혼합을 통한 최종 예측
#### 스태킹 앙상블 모델을 통한 회귀 예측
### 11. 정리
## 06 차원 축소
### 1. 차원 축소(Dimension Reduction) 개요
### 2. PCA(Principal Component Analysis)
#### PCA 개요
### 3. LDA(Linear Discriminant Analysis)
#### LDA 개요
#### 붓꽃 데이터 세트에 LDA 적용하기
### 4. SVD(Singular Value Decomposition)
#### SVD 개요
#### 사이킷런 TruncatedSVD 클래스를 이용한 변환
### 5. NMF(Non-Negative Matrix Factorization)
#### NMF 개요
### 6. 정리
## 07 군집화
### 1. K-군집화 알고리즘 이해
#### 사이킷런 KMeans 클래스 소개
