import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

date_range = pd.date_range(start = "2023-01-01", end = "2023-12-31", freq = "D")
n = len(date_range)
data = np.random.randn(n).cumsum()

ts_data = pd.DataFrame({"Date": date_range, "Value": data})
ts_data.set_index("Date", inplace = True)

plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data["Value"], label = "Time Series Data")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Sample Time Series Data")
plt.legend()
plt.grid(True)
plt.show()

decomposition = sm.tsa.seasonal_decompose(ts_data["Value"], model = "additive")
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize = (12, 8))
plt.subplot(411)
plt.plot(ts_data["Value"], label = "Original")
plt.legend(loc = "best")
plt.subplot(412)
plt.plot(trend, label = "Trend")
plt.legend(loc = "best")
plt.subplot(413)
plt.plot(seasonal, label = "Seasonality")
plt.legend(loc = "best")
plt.subplot(414)
plt.plot(residual, label = "Residuals")
plt.legend(loc = "best")
plt.tight_layout()
plt.show()
