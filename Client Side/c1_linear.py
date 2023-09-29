#client - 1 side

import tensorflow as tf
import pandas as pd

tf.random.set_seed(1)
# Defining the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(11,))
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Loading dataset
client = 1
num_data = 10000
df = pd.read_csv("./cardio_train.csv", sep=";")
# df = df[(client-1)*10000:client*10000]
Y = df['cardio']
X = df.drop(columns=['cardio','id'], axis=1)
# X["weight"] = X['weight'].astype(int)
x_train = X[(client-1)*10000:client*10000]
x_test = X[50000:]
y_train = Y[(client-1)*10000:client*10000]
y_test = Y[50000:]

r = model.fit(x_train);
print(r.history)