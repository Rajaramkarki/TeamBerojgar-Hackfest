#prediction side model

import numpy as np
import tensorflow as tf

# Base Model architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(11,))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

# Selecting the round with highest accuracy
round = 10
# Loading the weight
weights = np.load(f"round-{round}-weights.npz", allow_pickle=True)
weights = [weights['arr_%d' % i] for i in range(len(weights.files))]

# Creating the model
model = create_model()

# Setting weights to the base model
model.set_weights(weights)

# Predicting values
predictions = model.predict([[1893,2,168,62.0,110,80,1,1,0,0,1]])

print(predictions)