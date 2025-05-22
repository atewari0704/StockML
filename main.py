import streamlit as st
from datetime import date
import pandas as pd # Added for pd.DataFrame() and general data handling

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import requests
# import json # json is imported by requests, not strictly needed here unless used directly

# --- Configuration ---
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# --- Streamlit UI Elements ---
st.title("Stock Prediction App 📈")

user_input = st.text_input("Type the Stock Ticker (e.g., MSFT, GOOG) or Company Name for prediction").upper()
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365


# --- Functions ---
def getTicker(company_name_or_ticker):
    """
    Tries to find a stock symbol using Yahoo Finance's search API.
    This is particularly useful if the user enters a company name instead of a ticker.
    """
    st.info(f"Searching for ticker symbol for: {company_name_or_ticker}...")
    yfinance_search_url = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name_or_ticker, "quotes_count": 1, "country": "United States"}

    try:
        res = requests.get(url=yfinance_search_url, params=params, headers={'User-Agent': user_agent})
        res.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = res.json()
        
        if 'quotes' in data and data['quotes'] and data['quotes'][0]['symbol']:
            company_code = data['quotes'][0]['symbol']
            # Sometimes the search returns the input if it's already a ticker.
            # Or it might return a more "official" ticker.
            if company_code.upper() != company_name_or_ticker.upper():
                 st.success(f"Found symbol '{company_code}' for '{company_name_or_ticker}'.")
            else:
                st.info(f"Search confirmed symbol '{company_code}'.")
            return company_code
        else:
            st.warning(f"Could not find a ticker for '{company_name_or_ticker}' via Yahoo Finance search. Response did not contain a clear symbol.")
            return None # Return None if no valid symbol found
    except requests.exceptions.Timeout:
        st.error(f"Timeout occurred while trying to find ticker for '{company_name_or_ticker}'.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error while trying to find ticker for '{company_name_or_ticker}': {e}")
        return None
    except (KeyError, IndexError, TypeError) as e: # TypeError for res.json() if not json
        st.error(f"Error parsing ticker search results for '{company_name_or_ticker}': {e}")
        return None
    except Exception as e: 
        st.error(f"An unexpected error occurred in getTicker for '{company_name_or_ticker}': {e}")
        return None

def load_data(ticker_or_name, attempt=0, max_attempts=3, initial_input=None):
    """
    Loads historical stock data using yfinance.
    Includes logic to attempt ticker lookup if initial download fails,
    and prevents infinite recursion.
    """
    if not ticker_or_name:
        st.error("No ticker or company name provided to load_data.")
        return pd.DataFrame()

    if initial_input is None: # Store the very first user input to avoid re-looking up a resolved ticker
        initial_input = ticker_or_name

    if attempt >= max_attempts:
        st.error(f"Failed to load data for '{initial_input}' after {max_attempts} attempts. Please check the input or try again later.")
        return pd.DataFrame()

    st.info(f"Attempting to load data for: '{ticker_or_name}' (Attempt {attempt + 1}/{max_attempts})")
    
    # Try downloading data directly with the current ticker_or_name
    data = yf.download(ticker_or_name, start=START, end=TODAY, progress=False) # progress=False to avoid console prints
    
    if data.empty:
        st.warning(f"No data downloaded for '{ticker_or_name}'. This could be an invalid ticker, delisted stock, or no data for the period.")
        
        # Only try getTicker if it's not the last attempt and if current ticker_or_name might be a company name
        # or if it's the first attempt with user_input
        if attempt < max_attempts -1:
            # If the current ticker_or_name is the same as the initial_input, it means direct download failed,
            # so we try to resolve it as a company name.
            # If it's different, it means a previous getTicker result failed, so don't call getTicker again with it.
            if ticker_or_name.upper() == initial_input.upper() or attempt == 0 :
                st.info(f"Trying to find an alternative symbol for '{initial_input}' using lookup...")
                resolved_symbol = getTicker(initial_input) # Always try to resolve the original input
                
                if resolved_symbol and resolved_symbol.upper() != ticker_or_name.upper():
                    # If a NEW, different symbol is found, try loading data for it
                    return load_data(resolved_symbol, attempt + 1, max_attempts, initial_input)
                elif resolved_symbol: 
                    # If getTicker returns the same symbol that just failed OR a symbol that's already been tried implicitly
                    st.error(f"Ticker lookup for '{initial_input}' returned '{resolved_symbol}', which has already been tried or yielded no data. Stopping this path.")
                    return pd.DataFrame()
                else:
                    # getTicker failed to find any symbol
                    st.error(f"Could not resolve '{initial_input}' to a valid symbol after lookup.")
                    return pd.DataFrame()
            else:
                # This path means a resolved symbol (different from initial_input) also failed to load data.
                st.error(f"Previously resolved symbol '{ticker_or_name}' also failed to load data.")
                return pd.DataFrame()
        else:
            st.error(f"Reached max attempts for '{initial_input}', and no data was found.")
            return pd.DataFrame()
    else:
        data.reset_index(inplace=True)
        st.success(f"Successfully loaded data for '{ticker_or_name}'.")
        return data

# --- Main Application Logic ---
if user_input:  # Only proceed if user has entered something
    try:
        # Pass user_input as initial_input to load_data
        stock_data = load_data(user_input, initial_input=user_input)

        if not stock_data.empty:
            st.subheader(f'Raw data for {user_input}')
            st.write(stock_data.tail())

            # Plot raw data
            raw_fig = go.Figure()
            raw_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Open"], name="Stock Open", line_color="dodgerblue"))
            raw_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"], name="Stock Close", line_color="tomato", opacity=0.8))
            raw_fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
            st.plotly_chart(raw_fig)

            # Forecasting
            st.subheader("Forecasting")
            df_train = stock_data[["Date", "Close"]].copy() # Use .copy() to avoid SettingWithCopyWarning
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

            if len(df_train) < 10:  # Prophet generally needs more than just 2 points for a reasonable fit
                st.warning(f"Not enough historical data (found {len(df_train)} points) for '{user_input}' to make a reliable forecast. Prophet may struggle.")
            
            if len(df_train) >= 2: # Minimum for Prophet to run
                m = Prophet()
                with st.spinner('Fitting the forecast model...'):
                    m.fit(df_train)
                
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)

                st.subheader("Forecast data")
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                st.subheader("Forecast plot")
                fig1 = plot_plotly(m, forecast)
                st.plotly_chart(fig1)

                st.subheader("Forecast components")
                # plot_components returns a Matplotlib figure
                fig2 = m.plot_components(forecast)
                st.pyplot(fig2)
            else:
                 st.error(f"Cannot proceed with forecasting for '{user_input}' due to insufficient data (less than 2 data points).")

        else:
            # load_data function should have already shown specific errors.
            # This is a fallback message if data is empty after all attempts.
            st.error(f"Could not load any data for '{user_input}'. Please ensure the ticker/company name is correct and try again.")

    except ValueError as ve:
        st.error(f"A ValueError occurred: {ve}")
        st.error("This might be due to issues with the data format for Prophet or other numerical problems. Please check the input and data.")
    except Exception as e:
        st.error(f"An unexpected error occurred in the main application: {e}")
        st.exception(e) # This will print the full traceback in the Streamlit app for debugging
else:
    st.info("Please enter a stock ticker or company name to begin analysis.")


# import streamlit as st
# from  datetime import date

# import yfinance as yf
# from prophet import Prophet
# from prophet.plot import plot_plotly
# from plotly import graph_objs as go
# import requests
# import json 


# START = "2015-01-01"
# TODAY = date.today().strftime("%Y-%m-%d")

# st.title("Stock Prediction App 📈 ")

# user_input = st.text_input("Type the Stock Ticker for prediction").upper()

# n_years = st.slider("Years of prediction:",1,4)
# period = n_years * 365

# def getTicker(company_name):
#     yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
#     user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
#     params = {"q": company_name, "quotes_count": 1, "country": "United States"}

#     res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
#     data = res.json()

#     company_code = data['quotes'][0]['symbol']
#     return company_code


# def load_data(ticker):
#     data = yf.download(ticker, START, TODAY)
#     if data.empty:
#         symbol = getTicker(ticker.upper())
#         st.write(f"ticker is {symbol}")
#         return load_data(symbol)
#     else:
#         data.reset_index(inplace=True)
#         return data
    
# try:
#     data = load_data(user_input)

#     st.subheader('Raw data')
#     st.write(data.tail())

#     def plot_raw_data():
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=data["Date"],y=data["Open"],name="Stock Open",line_color="light blue"))
#         fig.add_trace(go.Scatter(x=data["Date"],y=data["Close"],name="Stock Close",line_color="red",opacity=0.5))
#         fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
#         st.plotly_chart(fig)

#     plot_raw_data()

#     #Forecasting
#     df_train = data[["Date","Close"]]
#     df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

#     m = Prophet()
#     m.fit(df_train)
#     future = m.make_future_dataframe(periods=period)
#     forecast = m.predict(future)

#     st.subheader("Forecast data")
#     st.write(forecast.tail())

#     st.write("forcast data")
#     fig1 = plot_plotly(m,forecast)
#     st.plotly_chart(fig1)

#     st.write("forecast components")
#     fig2 = m.plot_components(forecast)
#     st.write(fig2)


# except ValueError:
#     pass

