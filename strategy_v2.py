import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from simple_gsheets import push_to_sheets
except ModuleNotFoundError:
    push_to_sheets = None

ALL_TRADES: list[dict] = []
ALL_SUMMARIES: list[dict] = []

class RSIDMACrossover:
    # The `RSIDMACrossover` class implements a trading strategy based on RSI and moving average
    # crossovers, recording trades and calculating equity curve and metrics.
    def __init__(
        self,
        df: pd.DataFrame,
        ticker: str,
        rsi_period: int = 14,
        short_ma: int = 20,
        long_ma: int = 50,
        initial_capital: float = 1_000_000,
    ):
        self.df = df.copy()
        self.ticker = ticker
        self.rsi_period = rsi_period
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.initial_capital = initial_capital

        self.trades = []
        self._prepare_indicators()

    def _prepare_indicators(self) -> None:
        """
        The function calculates the Relative Strength Index (RSI) and simple moving averages (SMA) for a
        given DataFrame of stock data.
        """
        delta = self.df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        self.df["RSI"] = 100 - (100 / (1 + rs))
        self.df["SMA_S"] = self.df["Close"].rolling(self.short_ma).mean()
        self.df["SMA_L"] = self.df["Close"].rolling(self.long_ma).mean()

    def run(self, months: int = 6):
        """
        This Python function implements a trading strategy based on RSI and moving average crossovers,
        recording trades and calculating equity curve and metrics.
        
        :param months: The `months` parameter in the `run` method specifies the number of months of data to
        consider for the trading strategy. By default, it is set to 6 months if no value is provided when
        calling the method. This parameter is used to calculate the starting point for the analysis based on
        the, defaults to 6
        :type months: int (optional)
        :return: The `run` method is returning the instance of the class itself after performing the trading
        strategy logic, calculating metrics, and updating attributes like equity and final value.
        Additionally, it appends a summary of the trading results to a global list `ALL_SUMMARIES` for
        tracking purposes.
        """
        start = self.df.index.max() - pd.DateOffset(months=months)
        data = (
            self.df[self.df.index >= start]
            .dropna(subset=["RSI", "SMA_S", "SMA_L"])
            .copy()
        )

        cash = self.initial_capital
        position = 0
        rsi_armed = False
        equity_curve = []

        for i in range(1, len(data)):
            prev = data.iloc[i - 1]
            row = data.iloc[i]

            if row.RSI < 30:
                rsi_armed = True

            cross_up = prev.SMA_S <= prev.SMA_L and row.SMA_S > row.SMA_L
            if rsi_armed and cross_up and position == 0:
                units = int(cash // row.Close)
                if units:
                    cash -= units * row.Close
                    position = units
                    self._record_trade("BUY", row, units, cash, position)
                    rsi_armed = False

            if position and row.RSI > 70:
                cash += position * row.Close
                self._record_trade("SELL", row, position, cash, 0)
                position = 0

            equity_curve.append(cash + position * row.Close)

        self.equity = pd.Series(equity_curve, index=data.index[1:], name="Equity")
        self._calc_metrics()
        self.final_value = self.equity.iloc[-1]

        ALL_SUMMARIES.append(self.summary() | {"Ticker": self.ticker})
        return self

    def _record_trade(self, side: str, row: pd.Series, units: int, cash: float, position: int):
        """
        The `_record_trade` function appends trade details to a list and records trade information in a
        dictionary.
        
        :param side: The `side` parameter in the `_record_trade` method is a string that represents the side
        of the trade, such as 'Buy' or 'Sell'
        :type side: str
        :param row: The `row` parameter is a pandas Series object representing a row of data from a
        DataFrame. It seems to contain information related to a trading activity, such as the closing price
        of a stock at a specific time
        :type row: pd.Series
        :param units: The `units` parameter in the `_record_trade` method represents the number of units of
        a security that were traded in a particular transaction. It indicates the quantity of the security
        that was bought or sold during the trade
        :type units: int
        :param cash: The `cash` parameter in the `_record_trade` method represents the amount of cash
        involved in the trade. It is a float value that indicates the cash value associated with the trade
        being recorded
        :type cash: float
        :param position: The `position` parameter in the `_record_trade` method represents the current
        position of the trade. It is an integer value that indicates the number of shares or contracts held
        for a particular asset at the time of the trade. This parameter is used to calculate the total
        equity after the trade based on the
        :type position: int
        """
        self.trades.append((side, row.name, units, row.Close))
        ALL_TRADES.append({
            "Time":   row.name.strftime("%Y-%m-%d"),
            "Ticker": self.ticker,
            "Side":   side,
            "Price":  round(row.Close, 2),
            "Units":  units,
            "Cash":   round(cash, 2),
            "Position": position,
            "Equity": round(cash + position * row.Close, 2),
        })

    def _calc_metrics(self):
        """
        The `_calc_metrics` function calculates various trading metrics such as total return, max drawdown,
        and win rate based on the given trades and equity data.
        """
        self.total_return = self.equity.iloc[-1] / self.initial_capital - 1
        drawdown = self.equity.cummax() - self.equity
        self.max_drawdown = drawdown.max()
        sells = [t for t in self.trades if t[0] == "SELL"]
        buys = [t for t in self.trades if t[0] == "BUY"]
        wins = sum(s[3] > b[3] for b, s in zip(buys, sells))
        self.win_rate = wins / len(sells) if sells else np.nan

    def summary(self) -> dict:
        """
        This Python function returns a dictionary containing summary statistics related to trading
        performance.
        :return: A dictionary containing the following key-value pairs:
        - "return_%": The total return multiplied by 100 and rounded to 2 decimal places
        - "max_dd": The maximum drawdown rounded to 2 decimal places
        - "win_rate_%": The win rate multiplied by 100 and rounded to 2 decimal places, or "N/A" if the win
        rate is NaN
        - "
        """
        return {
            "return_%":   round(self.total_return * 100, 2),
            "max_dd":     round(self.max_drawdown, 2),
            "win_rate_%": round(self.win_rate * 100, 2) if not np.isnan(self.win_rate) else "N/A",
            "trades":     len(self.trades) // 2,
        }

def batch_backtest(csv_dir: str | Path = "data") -> pd.DataFrame:
    """
    This function performs a batch backtest on multiple CSV files in a specified directory and returns a
    DataFrame of the results.
    
    :param csv_dir: The `csv_dir` parameter is a string or a `Path` object that specifies the directory
    where the CSV files are located. If no directory is provided, it defaults to "data", defaults to
    data
    :type csv_dir: str | Path (optional)
    :return: The function `batch_backtest` is returning a pandas DataFrame that contains the results of
    running a trading strategy (RSIDMACrossover) on multiple CSV files located in the specified
    directory (`csv_dir`). The DataFrame has the results of the strategy for each CSV file as rows, with
    columns representing different metrics or summary information about the strategy's performance.
    """
    csv_dir = Path(csv_dir)
    results = {}
    for file in csv_dir.glob("*.csv"):
        df = pd.read_csv(file, parse_dates=["Date"], index_col="Date", skiprows=[1])
        strat = RSIDMACrossover(df, ticker=file.stem).run()
        results[file.stem] = strat.summary()
    return pd.DataFrame(results).T


if __name__ == "__main__":
    REPORT = batch_backtest("data")
    print("\n6 Month RSI + 20/50 DMA BackTest")
    print(REPORT.to_string())

    if push_to_sheets is not None:
        trades_df  = pd.DataFrame(ALL_TRADES)
        summary_df = pd.DataFrame(ALL_SUMMARIES).set_index("Ticker")
        push_to_sheets(trades_df, summary_df)
        print("✔ Google Sheet updated.")
