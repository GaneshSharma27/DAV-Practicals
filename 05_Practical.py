from pandas import read_csv
from pandas import to_datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# reading csv file
series = read_csv("shampoo_sales.csv", header = 0, index_col = 0, parse_dates = True, date_format = "%Y-%m").squeeze()

# converting index to period for monthly data
series.index = to_datetime("190" + series.index, format="%Y-%m").to_period("M")

# train-test split
x = series.values
size = int(len(x) * 0.66)
train, test = x[:size], x[size:]
history = [x for x in train]

# ARIMA model training and forecasting
predictions = []
for t in range(len(test)):
    model = ARIMA (history, order = (5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print(f"Predicted: {yhat:.3f}, Expected: {obs:.3f}")

# evaluate model
rmse = sqrt(mean_squared_error(test, predictions))
print(f"Test RMSE: {rmse:.3f}")

# plot results
pyplot.plot(test, label="Actual")
pyplot.plot(predictions, color="red", label="Predicted")
pyplot.legend()
pyplot.show()

