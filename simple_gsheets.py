from pathlib import Path
from typing import Union

import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd


CREDS_JSON: Union[str, Path] = "gsheets_key.json"
SHEET_NAME: str = "Trading-Log"


_SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _authorise() -> gspread.Spreadsheet:
    """
    The function `_authorise` returns a Google Sheets spreadsheet object after authorizing access using
    service account credentials.
    :return: The function `_authorise()` is returning a `gspread.Spreadsheet` object, which represents
    the Google Sheets spreadsheet that is being opened with the specified `SHEET_NAME`.
    """
    creds = ServiceAccountCredentials.from_json_keyfile_name(str(CREDS_JSON), _SCOPE)
    client = gspread.authorize(creds)
    return client.open(SHEET_NAME)


def _get_or_create(ws_title: str, sheet: gspread.Spreadsheet, ncols: int = 12):
    """Return a worksheet; create if it does not exist."""
    try:
        return sheet.worksheet(ws_title)
    except gspread.WorksheetNotFound:
        return sheet.add_worksheet(ws_title, rows="2000", cols=str(max(10, ncols)))


def push_to_sheets(trades_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """
    This function pushes trade and summary data to a Google Sheet after sorting and formatting the data
    appropriately.
    
    :param trades_df: The `trades_df` parameter is a pandas DataFrame containing trade data. It is
    sorted by the "Time" column before being pushed to a Google Sheet named "TradeLog"
    :type trades_df: pd.DataFrame
    :param summary_df: Summary_df is a DataFrame containing summary data related to trades. It is used
    to update a Google Sheet named "Summary" with the summary information
    :type summary_df: pd.DataFrame
    """
    sheet = _authorise()

    trades_df = trades_df.copy()
    trades_df.sort_values("Time", inplace=True)
    ws_log = _get_or_create("TradeLog", sheet, ncols=trades_df.shape[1])
    ws_log.clear()
    set_with_dataframe(ws_log, trades_df)

    sum_df = summary_df.reset_index()
    ws_sum = _get_or_create("Summary", sheet, ncols=sum_df.shape[1])
    ws_sum.clear()
    set_with_dataframe(ws_sum, sum_df)

    print("push_to_sheets(): Google Sheet updated.")
