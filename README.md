# 📈 Algo-Trading System with ML & Automation

This project is a **Python-based automated trading system** that fetches stock market data, applies a trading strategy (RSI + DMA crossover), runs a Logistic Regression model to predict price movement, and logs all results to **Google Sheets**.

---

## 🚀 Features

- ✅ Fetches daily stock data from Yahoo Finance
- ✅ Implements RSI (<30) + 20/50 DMA crossover trading strategy
- ✅ Backtests strategy over the last 6 months
- ✅ Trains a Logistic Regression classifier using technical indicators
- ✅ Outputs prediction accuracy for each stock
- ✅ Automatically pushes trades and summaries to Google Sheets
- ✅ Fully modular and ready for automation

---

## 📂 Project Structure

```
├── auto_run.py               # Main script to run the full pipeline
├── data_loader.py           # Functions to fetch and save stock data
├── model_log_reg_3.py       # ML model and feature engineering
├── strategy_v2.py           # RSI + DMA strategy logic and backtesting
├── simple_gsheets.py        # Google Sheets integration
├── requirements.txt         # Required packages
├── data/                    # Folder where stock CSVs are saved
├── models/                  # Trained logistic regression models
└── gsheets_key.json         # (You need this) Google Sheets API key
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/KayEssP-26/Algo_trading_HtoH.git
cd algo-trading-system
```

### 2. Create Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Setup Google Sheets

- Create a Google Service Account and download the `gsheets_key.json` file.
- Share your Google Sheet with the client email from the credentials.
- Rename the sheet as `Trading-Log`, or update the name in `simple_gsheets.py`.

---

## ▶️ How to Run

Run the entire pipeline:

```bash
python auto_run.py
```

This will:
1. Fetch fresh 5-year stock data
2. Train/test ML model for each stock
3. Run the RSI + 20/50 DMA crossover backtest
4. Log trades and summaries to Google Sheets

---

## 📊 Strategy Logic

**Buy Signal:**
- RSI < 30 → "Armed"
- Buy when 20-DMA crosses above 50-DMA while armed

**Sell Signal:**
- RSI > 70

The strategy is backtested over the last **6 months** of data. Metrics like total return, max drawdown, and win rate are calculated and logged.

---

## 🤖 Machine Learning

Uses **Logistic Regression** to predict next-day movement based on:
- RSI14
- MACD difference
- SMA slope
- Log return
- Volume z-score

Each model is trained and saved per stock, with accuracy reported for both train and test sets.

---

## 📈 Google Sheets Integration

Two tabs are created:
- **TradeLog** – detailed trades with price, units, and date
- **Summary** – return %, win rate, number of trades

Powered by `gspread` + `oauth2client`.

---


## 🙋‍♂️ Author

**Karan Singh** 
Karan.s.punni@gmail.com
