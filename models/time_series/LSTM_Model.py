import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/BDA_Project"
data_path = "stocks_data/final_stock_consolidated.csv"
data_path = os.path.join(root_dir, data_path)

df = pd.read_csv(data_path)
df.head()

df['Date'] = pd.to_datetime(df['Date'])
df.info()

df = df.sort_values(by='Date')
df.head()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

label_stock = label_encoder.fit_transform(df['stock_ticker'])
df['stock_ticker'] = label_stock
df.head()

df['Date'] = df['Date'].values.astype(np.int64)
df.head()
df = df.drop(df.columns[[2, 3, 4, 5, 6]], axis=1)

training_set = df.iloc[:2500000, ].values
test_set = df.iloc[2500000:, ].values
test_data = df.iloc[2500000:, ]
training_data = df.iloc[:2500000, ]

test_set_stock = test_data['Open'].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))

training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 2500000):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units=50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs=10, batch_size=64)

test_set_scaled = sc.fit_transform(test_set)
dataset_total = pd.concat((training_data, test_data), axis=0)

inputs = dataset_total[len(dataset_total) - len(test_data) - 60:].values
inputs = inputs.reshape(-1, 1)

inputs = sc.transform(inputs)
X_test = []
for i in range(60, 1026176):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)

sc.inverse_transform(predicted_stock_price[:, [0]])

test_set_stock1 = test_set_stock.reshape(-1, 1)

test_set_scaled = sc.fit_transform(test_set_stock1)

from sklearn.metrics import mean_squared_error

testScore = np.sqrt(mean_squared_error(test_set_scaled, predicted_stock_price))
print('Test Score: %.2f RMSE' % (testScore))
