import flwr as fl
import tensorflow as tf
import pandas as pd
import sys
import streamlit as st

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
Y = df['cardio']
X = df.drop(columns=['cardio','id'], axis=1)
x_train = X[(client-1)*10000:client*10000]
x_test = X[50000:]
y_train = Y[(client-1)*10000:client*10000]
y_test = Y[50000:]

# Create a Streamlit app
st.title("server Side Accuracy Data")

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=32, verbose=0)
        # hist = r.history
        # st.write("Fit history : " ,hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        st.write("Eval accuracy:", accuracy)  # Display accuracy using Streamlit
        return loss, len(x_test), {"accuracy": accuracy}

# Starting Flower client
st.write("Connecting to the server...")
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length=1024*1024*1024
)
