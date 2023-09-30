import flwr as fl
import tensorflow as tf
import sys
import pandas as pd
import streamlit as st

tf.random.set_seed(1)

# Defining the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(11,))
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Loading dataset
df = pd.read_csv("./client2.csv")
y_train = df['cardio']
x_train = df.drop(columns=['cardio','id'], axis=1)

df_test =  pd.read_csv("./test.csv")
y_test = df_test['cardio']
x_test = df_test.drop(columns=['cardio','id'], axis=1)

# Create a Streamlit app
st.markdown("<h1><b>Project DAML:</b></h1>", unsafe_allow_html=True)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=32, verbose=0)
        hist = r.history

        filtered_hist = {key: value for key, value in hist.items() if key in ["loss", "accuracy"]}

        # Display "Client-Side Training History:" in bold and larger size
        st.markdown("<h3><b>Client-Side Training History:</b></h3>", unsafe_allow_html=True)

        for key, values in filtered_hist.items():
            st.write(f"{key}: {values}")

            if key == "accuracy":
                st.write("Plotting Accuracy Values:")
                chart = st.line_chart(values)

        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        st.markdown("<h3><b>Global Server Accuracy:</b></h3>", unsafe_allow_html=True)
        st.write("Global accuracy:", accuracy)  # Display global server accuracy using Streamlit
        return loss, len(x_test), {"accuracy": accuracy}

# Starting Flower client
st.write("Connecting to the server...")
fl.client.start_numpy_client(
    # server_address="192.168.11.66:"+str(sys.argv[1]), 
    server_address="localhost:"+str(sys.argv[1]), 
    client=FlowerClient(), 
    grpc_max_message_length=1024*1024*1024
)
