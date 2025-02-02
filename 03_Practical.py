import pandas as pd
from sklearn import linear_model

df = pd.read_csv("cars.csv")

x = df[["Weight", "Volume"]]
y = df["CO2"]

regr = linear_model.LinearRegression()
regr.fit(x, y)

predicted_CO2 = regr.predict([[2300, 1300]])
print("Predicted CO2 [weight = 2300 kg, volume =  1300 ccm]:", predicted_CO2)

print("Coefficient [weight, volume]")
print(regr.coef_)
