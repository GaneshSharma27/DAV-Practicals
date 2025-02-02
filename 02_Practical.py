import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("salary_data.csv")
x = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

z = x * y
x2 = x * x

a = ((len(x) * sum(z)) - (sum(x) * sum(y))) / ((len(x) * sum(x2)) - (sum(x) * sum(x)))
b = (sum(y) - (a * sum(x))) / (len(x))

x_test = [1.7, 2.3, 3.9, 4.5, 5.4]
y_pred = []
for i in range(len(x_test)):
    y_pred.append((a * x_test[i]) + b)

regressor = LinearRegression()
regressor.fit(x.reshape(-1, 1), y)

plt.plot(x.reshape(-1, 1), regressor.predict(x.reshape(-1, 1)), color = "blue", label = "Best fit line")
plt.scatter(x, y, color = "green", label = "Actual points")
plt.scatter(x_test, y_pred, color = "red", label = "Predicted points")
plt.title("Salary v/s Experience")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")
plt.legend(loc = "upper left")
plt.show()

