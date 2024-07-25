import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie danych
data = pd.read_csv('sales_data.csv', parse_dates=['date'], index_col='date')

# Wyświetlenie pierwszych kilku wierszy danych
print(data.head())

# Normalizacja danych
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data['sales'] = scaler.fit_transform(data['sales'].values.reshape(-1, 1))

# Podział danych na zestawy treningowy i testowy
split_fraction = 0.8
train_size = int(len(data) * split_fraction)
train_data = data[:train_size]
test_data = data[train_size:]


_____________________

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 30  # na przykład użyjemy 30 dni jako długość sekwencji
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)


____________________________________

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')


_____________________________________________

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Rysowanie wykresu
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Rzeczywiste dane')
plt.plot(data.index[-len(predictions):], predictions, label='Prognozowane dane')
plt.xlabel('Data')
plt.ylabel('Sprzedaż')
plt.legend()
plt.show()
