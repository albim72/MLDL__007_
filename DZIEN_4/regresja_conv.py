import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Generowanie sztucznych danych czasowych
def generate_data(n_samples, n_features, n_timesteps):
    X = np.random.rand(n_samples, n_timesteps, n_features)
    y = np.sum(X, axis=1) + np.random.randn(n_samples, n_features) * 0.1  # Zakładamy, że przewidujemy sumę cech na przestrzeni czasu
    return X, y

n_samples = 1000
n_features = 5
n_timesteps = 10

X, y = generate_data(n_samples, n_features, n_timesteps)

# Normalizacja danych
scaler = MinMaxScaler()
X = X.reshape(-1, n_features)
X = scaler.fit_transform(X)
X = X.reshape(n_samples, n_timesteps, n_features)

y = scaler.fit_transform(y)

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_features))

model.compile(optimizer=Adam(), loss='mse')

model.summary()
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Loss: {loss}')
predictions = model.predict(X_test)
print(predictions[:5])
print(y_test[:5])
