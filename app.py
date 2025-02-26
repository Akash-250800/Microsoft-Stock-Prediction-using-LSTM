import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model
import yfinance as yf
import os

# Load Data
def load_data(ticker):
    df = yf.download(ticker, start="2010-01-01", end="2023-01-01")
    df = df[['Close']]
    return df

# Preprocess Data
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Create Dataset for LSTM
def create_dataset(data, time_step=100):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Build LSTM Model
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train Model
def train_model(model, X_train, y_train, epochs=10, batch_size=64):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Save Model
def save_model(model, filename='stock_lstm_model.h5'):
    model.save(filename)

# Predict
def predict_stock(model, data, time_step):
    predictions = []
    for i in range(time_step, len(data)):
        input_data = data[i - time_step:i, 0].reshape(1, time_step, 1)
        predictions.append(model.predict(input_data))
    return np.array(predictions).reshape(-1)

# Main Streamlit App
def main():
    st.title("Microsoft LSTM Stock Prediction")

    # Sidebar for input
    ticker = st.sidebar.text_input("Enter Stock Ticker", "MSFT")
    model_file = st.sidebar.file_uploader("Upload pre-trained model (.h5)", type=["h5"])

    # Load and display data
    st.write(f"Stock Price Data for {ticker}")
    df = load_data(ticker)
    st.line_chart(df['Close'])

    # Preprocess data
    data, scaler = preprocess_data(df)
    time_step = 100
    X, y = create_dataset(data, time_step)

    # Reshape data
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Load or train the model
    if model_file:
        model = load_model(model_file)
        st.success("Model Loaded!")
    else:
        model = build_model()
        model = train_model(model, X, y)
        save_model(model)
        st.success("Model Trained and Saved!")

    # Predict future prices
    predictions = predict_stock(model, data, time_step)
    st.write("Predicted Stock Prices")
    predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))
    st.line_chart(predicted_prices)

if __name__ == "__main__":
    main()