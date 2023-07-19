import streamlit as st
from  datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import requests
import json 


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App 📈 ")

user_input = st.text_input("Type the Stock Ticker for prediction").upper()

n_years = st.slider("Years of prediction:",1,4)
period = n_years * 365

def getTicker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    if data.empty:
        symbol = getTicker(ticker.upper())
        st.write(f"ticker is {symbol}")
        return load_data(symbol)
    else:
        data.reset_index(inplace=True)
        return data
    
try:
    data = load_data(user_input)

    st.subheader('Raw data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"],y=data["Open"],name="Stock Open",line_color="light blue"))
        fig.add_trace(go.Scatter(x=data["Date"],y=data["Close"],name="Stock Close",line_color="red",opacity=0.5))
        fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    #Forecasting
    df_train = data[["Date","Close"]]
    df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader("Forecast data")
    st.write(forecast.tail())

    st.write("forcast data")
    fig1 = plot_plotly(m,forecast)
    st.plotly_chart(fig1)

    st.write("forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)


except ValueError:
    pass

