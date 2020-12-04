import pandas as pd
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
y = [10, 20, 30, 40, 50, 60]
z = [8, 18, 28, 38, 48, 58]

x_df = pd.DataFrame(x, columns=['x'])
y_df = pd.DataFrame(y, columns=['y'])
z_df = pd.DataFrame(z, columns=['z'])

plt.plot(y_df, x_df)
plt.plot(z_df, x_df)
plt.show()
