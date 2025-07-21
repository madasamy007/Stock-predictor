import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

st.title("Stock Price Prediction using LSTM")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, RELIANCE.NS)", "AAPL")

if st.button("Predict"):
    st.info("Fetching data...")
    df = yf.download(ticker, start="2015-01-01", end="2023-12-31")

    if df.empty:
        st.error("❌ No data found for the given ticker symbol. Please try another.")
    else:
        df = df[["Close"]]
        df.dropna(inplace=True)

        if len(df) < 60:
            st.warning("⚠️ Not enough data to create training sequences (need at least 60 days).")
        else:
            # Normalize data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df)

            # Create sequences
            x_train, y_train = [], []
            for i in range(60, len(scaled_data)):
                x_train.append(scaled_data[i - 60:i])
                y_train.append(scaled_data[i])

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Build LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mean_squared_error")

            with st.spinner("Training model..."):
                model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

            # Predictions
            predicted = model.predict(x_train)
            predicted = scaler.inverse_transform(predicted)
            actual = scaler.inverse_transform(y_train.reshape(-1, 1))

            # Plot results
            st.subheader("📊 Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(actual, label="Actual", color="black")
            ax.plot(predicted, label="Predicted", color="green")
            ax.legend()
            st.pyplot(fig)
