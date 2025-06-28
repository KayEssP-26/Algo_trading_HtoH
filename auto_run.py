from data_loader import fetch_multiple_stocks, save_data_locally
from model_log_reg_3 import batch_run as run_ml
from strategy_v2 import batch_backtest, ALL_TRADES, ALL_SUMMARIES

try:
    from simple_gsheets import push_to_sheets
except ImportError:
    push_to_sheets = None

def run_pipeline():
    """
    This Python function runs a data pipeline that fetches stock data, runs a machine learning model,
    executes a trading strategy, and optionally pushes the results to Google Sheets.
    """
    # 1. Fetch updated data
    print("\nFetching fresh stock data...")
    stocks = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS']
    data = fetch_multiple_stocks(stocks, period='5y', interval='1d')
    save_data_locally(data)

    # 2. Run ML model
    print("\nRunning ML model...")
    ml_report = run_ml("data")
    print(ml_report.to_string())

    # 3. Run strategy
    print("\nRunning RSI + DMA backtest strategy...")
    strat_report = batch_backtest("data")
    print(strat_report.to_string())

    # 4. Push to Sheets
    if push_to_sheets is not None:
        print("\nPushing to Google Sheets...")
        import pandas as pd
        trades_df = pd.DataFrame(ALL_TRADES)
        summary_df = pd.DataFrame(ALL_SUMMARIES).set_index("Ticker")
        push_to_sheets(trades_df, summary_df)
    else:
        print("\nGoogle Sheets module not available or not configured.")

if __name__ == "__main__":
    run_pipeline()
