import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://raw.githubusercontent.com/dscoool/Machine-Learning-with-Python-and-Spark/refs/heads/master/Linear-Regression/Ecommerce_Customers.csv")

cdf = df[["Avg Session Length", "Time on App", "Time on Website", "Length of Membership", "Yearly Amount Spent"]]

plt.scatter(cdf["Length of Membership"], cdf["Yearly Amount Spent"], color='blue')
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train["Length of Membership"], train["Yearly Amount Spent"], color='blue')
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")
plt.show()

inputCols = ["Avg Session Length", "Time on App", "Time on Website", "Length of Membership"]
x_train = np.asanyarray(train[inputCols])
y_train = np.asanyarray(train[["Yearly Amount Spent"]])

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

print("Coefficients:", regr.coef_)

x_test = np.asanyarray(test[inputCols])
y_test = np.asanyarray(test[["Yearly Amount Spent"]])

y_hat = regr.predict(x_test)

mse = np.mean((y_hat - y_test) ** 2)
print("Residual sum of squares (MSE): %.2f" % mse)

variance_score = regr.score(x_test, y_test)
print("Variance score (R²): %.2f" % variance_score)
