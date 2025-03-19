# 라이브러리 불러오기
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

# 데이터 불러오기
df = pd.read_csv("https://raw.githubusercontent.com/dscoool/Machine-Learning-with-Python-and-Spark/refs/heads/master/Linear-Regression/Ecommerce_Customers.csv")

# 필요한 컬럼 선택
cdf = df[["Avg Session Length", "Time on App", "Time on Website", "Length of Membership", "Yearly Amount Spent"]]

# 시각화: Length of Membership vs Yearly Amount Spent
plt.scatter(cdf["Length of Membership"], cdf["Yearly Amount Spent"], color='blue')
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")
plt.show()

# 데이터 분할 (80% 훈련, 20% 테스트)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# 훈련 데이터 시각화
plt.scatter(train["Length of Membership"], train["Yearly Amount Spent"], color='blue')
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")
plt.show()

# 입력 변수 선택
inputCols = ["Avg Session Length", "Time on App", "Time on Website", "Length of Membership"]
x_train = np.asanyarray(train[inputCols])
y_train = np.asanyarray(train[["Yearly Amount Spent"]])

# 모델 학습
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

# 모델 계수 출력
print("Coefficients:", regr.coef_)

# 예측 수행
x_test = np.asanyarray(test[inputCols])
y_test = np.asanyarray(test[["Yearly Amount Spent"]])

y_hat = regr.predict(x_test)

# 잔차 제곱합 (MSE) 출력
mse = np.mean((y_hat - y_test) ** 2)
print("Residual sum of squares (MSE): %.2f" % mse)

# 모델 평가 (R² 점수)
variance_score = regr.score(x_test, y_test)
print("Variance score (R²): %.2f" % variance_score)
