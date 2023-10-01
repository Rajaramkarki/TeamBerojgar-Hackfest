#prediction side model

import numpy as np
import tensorflow as tf

# Base Model architecture
def create_model():
    tf.random.set_seed(1)

    # Learning rate
    learning_rate = 0.02

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(11,)),
        tf.keras.layers.Dropout(0.6),  # Dropout layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Use learning rate in Adam optimizer
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=adam_optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
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
