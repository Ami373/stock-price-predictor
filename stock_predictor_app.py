import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from streamlit_autorefresh import st_autorefresh
import google.generativeai as genai
import os

st.set_page_config(page_title="Stock Price Predictor & Dashboard", layout="wide")

# Auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="data_refresh")

st.title("📈 Stock Price Predictor & Dashboard")

symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, INFY.BO)", value='AAPL')
days = st.sidebar.slider("Days of Historical Data", 100, 1000, 365)
predict_days = st.sidebar.slider("Days to Predict Ahead", 1, 30, 7)

@st.cache_data(show_spinner=False)
def load_data(symbol, days):
    df = yf.download(symbol, period=f"{days}d")
    return df

def prepare_data(df, predict_days):
    df_filtered = df[['Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_filtered)

    X, y = [], []
    for i in range(60, len(scaled_data) - predict_days):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i:i+predict_days, 0])

    return np.array(X), np.array(y), scaler

def build_model(input_shape, output_days):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=output_days))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Gemini News Sentiment Analysis
@st.cache_data(show_spinner=False)
def get_sentiment(company):
    try:
        gemini_api_key = st.secrets["GEMINI_NEWS_API_KEY"]
    except Exception:
        return [("Missing or invalid Gemini News API key.", 0)]

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

        prompt = f"""
        Give me 3 recent news headlines related to the stock performance of {company}.
        For each, also label the sentiment as Positive, Negative, or Neutral.
        Example format:
        - Headline — Sentiment: Positive
        """

        response = model.generate_content(prompt)
        text = response.text

        lines = text.split("\n")
        sentiments = []
        for line in lines:
            if not line.strip():
                continue
            polarity = TextBlob(line).sentiment.polarity
            sentiments.append((line.strip(), polarity))
        return sentiments

    except Exception as e:
        return [(f"Error using Gemini API: {e}", 0)]

try:
    df = load_data(symbol, days)
    if df.empty:
        st.error("Failed to load data. Check stock symbol.")
    else:
        st.subheader(f"📊 Historical Data for {symbol}")
        st.line_chart(df['Close'])

        X, y, scaler = prepare_data(df, predict_days)

        if X.shape[0] == 0:
            st.warning("Not enough data for training the model. Try reducing prediction days or increasing historical days.")
        else:
            X = X.reshape(X.shape[0], X.shape[1], 1)

            model = build_model((X.shape[1], 1), predict_days)
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            last_60_days = df[['Close']].values[-60:]
            if last_60_days.shape[0] < 60:
                st.warning("Not enough data to make prediction. Try increasing the number of historical days.")
                prediction = []
            else:
                last_60_scaled = scaler.transform(last_60_days)
                X_predict = last_60_scaled.reshape(1, 60, 1)
                prediction_scaled = model.predict(X_predict)
                prediction = scaler.inverse_transform(prediction_scaled).flatten()

            st.subheader("📉 Prediction for Next Days")
            st.line_chart(prediction)

            # Investment suggestion
            if len(prediction) > 0:
                diff = float(prediction[-1]) - float(df['Close'].iloc[-1])
                suggestion = "Buy" if diff > 0 else "Sell" if diff < 0 else "Hold"
                st.subheader("💡 Investment Suggestion")
                st.success(f"Suggested Action: {suggestion}")

            # News Sentiment
            st.subheader("📰 News Sentiment Analysis")
            news_sentiments = get_sentiment(symbol)
            for title, polarity in news_sentiments:
                sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
                st.write(f"- {title} — *{sentiment}*")

            # Portfolio Tracker
            st.sidebar.subheader("💼 Portfolio Tracker")
            tickers = st.sidebar.text_input("Track Multiple Stocks (comma separated)", "AAPL,MSFT")
            tickers = [t.strip().upper() for t in tickers.split(',') if t.strip() != '']

            st.subheader("💰 Portfolio Value")
            total_value = 0
            for ticker in tickers:
                data = yf.download(ticker, period="1d", interval="1m")
                if not data.empty:
                    try:
                        latest_price = float(data['Close'].iloc[-1])
                        qty = st.sidebar.number_input(f"{ticker} Quantity", min_value=0.0, value=0.0, step=0.1, format="%.2f", key=f"qty_{ticker}")
                        value = latest_price * qty
                        st.write(f"{ticker}: ${latest_price:.2f} × {qty} = ${value:.2f}")
                        total_value += value
                    except Exception as e:
                        st.warning(f"Failed to process {ticker}: {e}")
            st.success(f"Total Portfolio Value: ${total_value:.2f}")

except Exception as e:
    st.error(f"Error loading or analyzing data: {e}")