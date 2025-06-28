import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

def rsi(series: pd.Series, period: int = 14):
    """
    The function calculates the Relative Strength Index (RSI) for a given series using a specified
    period.
    
    :param series: The `series` parameter is expected to be a pandas Series containing the data for
    which you want to calculate the Relative Strength Index (RSI). This could be a series of stock
    prices, index values, or any other numerical data that you want to analyze for momentum
    :type series: pd.Series
    :param period: The `period` parameter in the `rsi` function represents the time period over which
    the Relative Strength Index (RSI) is calculated. By default, it is set to 14, but you can adjust
    this parameter to calculate the RSI over a different period if needed, defaults to 14
    :type period: int (optional)
    :return: The function `rsi` calculates the Relative Strength Index (RSI) for a given pandas Series
    `series` with an optional period parameter. It calculates the RSI value based on the series data and
    returns the RSI values for each data point in the series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    The function calculates the Moving Average Convergence Divergence (MACD) line and signal line for a
    given time series data.
    
    :param series: The `series` parameter is expected to be a pandas Series containing the data for
    which you want to calculate the Moving Average Convergence Divergence (MACD)
    :type series: pd.Series
    :param fast: The `fast`, `slow`, and `signal` parameters are commonly used in calculating the Moving
    Average Convergence Divergence (MACD) indicator in technical analysis, defaults to 12
    :type fast: int (optional)
    :param slow: The `slow` parameter in the Moving Average Convergence Divergence (MACD) calculation
    represents the number of periods used to calculate the slower Exponential Moving Average (EMA). In
    the provided function, the `slow` parameter is set to a default value of 26 if not specified when
    calling, defaults to 26
    :type slow: int (optional)
    :param signal: The `signal` parameter in the Moving Average Convergence Divergence (MACD)
    calculation represents the period for the signal line. It is the number of periods used to calculate
    the moving average of the MACD line to generate the signal line. In the provided function, the
    default value for the, defaults to 9
    :type signal: int (optional)
    :return: The `macd` function is returning two pandas Series objects - `macd_line` and `macd_signal`.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal


def make_features(df: pd.DataFrame):
    """
    The function `make_features` calculates various technical indicators and features for a given
    DataFrame of financial data.
    
    :param df: The function `make_features` takes a pandas DataFrame `df` as input and performs various
    feature engineering tasks on it to create new columns. Here is a summary of the features that are
    being created:
    :type df: pd.DataFrame
    :return: The function `make_features` is returning a modified DataFrame `df` with additional
    columns/features added such as RSI14, MACD_diff, SMA20, SMA50, SMA_slope, LogRet, Vol_z, and y. The
    function calculates these features based on the existing columns in the input DataFrame `df` and
    returns the modified DataFrame after dropping any rows with missing values.
    """
    df = df.copy()

    df["RSI14"] = rsi(df["Close"])
    macd_line, macd_signal = macd(df["Close"])

    df["MACD_diff"] = macd_line - macd_signal
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA_slope"] = (df["SMA20"] - df["SMA50"]) / df["Close"]
    df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

    df["y"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df

def fit_logreg(train_X, train_y):
    """
    The function `fit_logreg` fits a logistic regression model with balanced class weights using
    standardized features.
    
    :param train_X: It looks like the code snippet you provided is a function `fit_logreg` that fits a
    logistic regression model using the given training data `train_X` and `train_y`. The function uses a
    pipeline with a StandardScaler for feature scaling and a LogisticRegression classifier with specific
    hyperparameters
    :param train_y: It looks like the definition of the `fit_logreg` function is missing the `train_y`
    parameter. The `train_y` parameter should be the target variable for the logistic regression model.
    It is the variable that the model will try to predict based on the features in `train_X`
    :return: The function `fit_logreg` returns a pipeline object that consists of a StandardScaler and a
    LogisticRegression classifier with specified parameters. The pipeline is fitted on the training data
    (`train_X` and `train_y`) before being returned.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(train_X, train_y)
    return pipe

def run_one_ticker(csv_path: Path, test_size: float = 0.2):
    """
    This function reads stock data from a CSV file, preprocesses it, trains a logistic regression model,
    evaluates its accuracy on training and testing data, and saves the model.
    
    :param csv_path: The `csv_path` parameter is the file path to the CSV file containing the data for a
    particular stock ticker. This function reads the data from the CSV file, preprocesses it by creating
    additional features, splits the data into training and testing sets, fits a logistic regression
    model, evaluates the model's
    :type csv_path: Path
    :param test_size: The `test_size` parameter in the `run_one_ticker` function is used to specify the
    proportion of the dataset that should be used for testing the model. By default, it is set to 0.2,
    which means 20% of the data will be used for testing and the
    :type test_size: float
    :return: The function `run_one_ticker` returns a dictionary containing the following information:
    - "ticker": The stem of the CSV file path
    - "train_acc": The rounded training accuracy score of the logistic regression model
    - "test_acc": The rounded testing accuracy score of the logistic regression model
    - "train_size": The number of samples in the training set
    - "test_size": The
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date", skiprows=[1])
    df = make_features(df)
    feature_cols = ["RSI14", "MACD_diff", "SMA_slope", "LogRet", "Vol_z"]
    X = df[feature_cols]
    y = df["y"]

    split = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = fit_logreg(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, f"models/{csv_path.stem}_logreg.pkl")

    return {
        "ticker": csv_path.stem,
        "train_acc": round(acc_train, 3),
        "test_acc": round(acc_test, 3),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

def batch_run(data_dir: str | Path = "data"):
    """
    The function `batch_run` processes multiple CSV files in a specified directory and returns a
    DataFrame with the results indexed by ticker.
    
    :param data_dir: The `data_dir` parameter is a string or a `Path` object that specifies the
    directory where the CSV files are located. If no directory is provided, it defaults to "data",
    defaults to data
    :type data_dir: str | Path (optional)
    :return: The function `batch_run` returns a pandas DataFrame with the records from running
    `run_one_ticker` on each CSV file in the specified `data_dir`. The DataFrame is then set to be
    indexed by the "ticker" column.
    """
    records = []
    for csv_file in Path(data_dir).glob("*.csv"):
        rec = run_one_ticker(csv_file)
        records.append(rec)
    return pd.DataFrame(records).set_index("ticker")

if __name__ == "__main__":
    report = batch_run("data")
    print(report.to_string())
