# Getting introduced to data analytics libraries in Python and R

# ------------------ NumPy ------------------
import numpy as np
a = np.array([5, 8, 12])
print(a)

# ------------------ Scikit-learn ------------------
from sklearn import linear_model
lin_reg = linear_model.LinearRegression()
lin_reg.fit([[0,0], [2, 2], [4, 4]], [0, 2, 4])
print(lin_reg.coef_)

# ------------------ Pandas ------------------
import pandas as pd
import numpy as np
info = np.array(["P", "a", "n", "d", "a", "s"])
b = pd.Series(info)
print(b)

x = ["Python", "Pandas"]
df = pd.DataFrame(x)
print(df)

# ------------------ Matplotlib ------------------
import matplotlib.pyplot as plt
import numpy as np

coordinate = np.linspace(0, 10, 100)
plt.plot(x, x, label = "linear")
plt.legend()
plt.show()

# ------------------ SciPy ------------------
from scipy.signal import chirp as cp
from scipy.signal import spectrogram as sp
import matplotlib.pyplot as plt
import numpy as np

t_T = np.linspace(3, 10, 300)
w_W = cp(t_T, f0 = 4, f1 = 2, t1 = 5, method = "linear")
plt.plot(t_T, w_W)
plt.title("Linear Chirp")
plt.xlabel("Time in seconds")
plt.show()
