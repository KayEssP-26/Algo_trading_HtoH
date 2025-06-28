import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_STOCKS = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS']

def fetch_stock_data(symbol, period='5y', interval='1d'):
    """
    Fetches historical stock data for a given symbol using yfinance.
    """
    try:
        logging.info(f"Fetching data for {symbol}...")
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_multiple_stocks(symbols=DEFAULT_STOCKS, period='5y', interval='1d'):
    """
    Fetches historical data for multiple stocks and returns a dictionary of DataFrames.
    """
    data_dict = {}
    for symbol in symbols:
        df = fetch_stock_data(symbol, period, interval)
        if not df.empty:
            data_dict[symbol] = df
    return data_dict

def save_data_locally(data_dict, folder='data'):
    """
    Saves each stock's data as a CSV file.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    for symbol, df in data_dict.items():
        filepath = os.path.join(folder, f"{symbol.replace('.NS', '')}.csv")
        df.to_csv(filepath, index=False)
        logging.info(f"Saved {symbol} data to {filepath}")

if __name__ == "__main__":
    stocks = DEFAULT_STOCKS
    data = fetch_multiple_stocks(stocks)
    save_data_locally(data)
