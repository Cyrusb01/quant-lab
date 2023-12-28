import pandas as pd
import vectorbtpro as vbt
import os
import datetime
import pytz


def update_csv_data():
    assets = ["BTCUSDT", "ETHUSDT"]

    for asset in assets:
        file = f"data/{asset}.csv"

        btc_df = pd.read_csv(file, index_col=0, parse_dates=True)
        # Now convert the dataframe into a BinanceData object
        btcdata = vbt.BinanceData.from_data(vbt.symbol_dict({asset: btc_df}))
        print(btcdata.last_index)
        # Now update the data
        btcdata = btcdata.update(end="now UTC", timeframe="1m")
        print(btcdata.last_index)

        # save as csv
        df_data = btcdata.get()
        df_data = df_data[["Open", "High", "Low", "Close", "Volume"]]
        df_data = df_data.reset_index()
        df_data.rename(columns={"index": "datetime"}, inplace=True)
        df_data["datetime"] = pd.to_datetime(df_data["datetime"])
        df_data = df_data.set_index("datetime")
        df_data.to_csv(f"data/{asset}.csv")


def fetch_binance_data(ASSET):
    
    data_path = f"data/{ASSET}_futures.pkl"

    start_date = datetime.datetime(2018, 12, 31, tzinfo=pytz.utc)
    freq = "1m"
    binance_data = vbt.BinanceData.fetch(
        ASSET, timeframe=freq, klines_type=2, start=start_date, end="now UTC"
    )
    # binance_data = vbt.BinanceData.fetch(ASSET,timeframe=freq, start=start_date,end=end_date)
    binance_data.save(data_path)
    print("Downloaded Data")

    # load the data
    binance_data = vbt.BinanceData.load(data_path)
    df_data = binance_data.get()

    df_data = df_data[["Open", "High", "Low", "Close", "Volume"]]
    df_data = df_data.reset_index()
    df_data.rename(columns={"Open time": "datetime"}, inplace=True)
    df_data["datetime"] = pd.to_datetime(df_data["datetime"])
    df_data = df_data.set_index("datetime")
    return df_data

def pickle_to_df(data_path):
    # load the data
    binance_data = vbt.BinanceData.load(data_path)
    df_data = binance_data.get()

    df_data = df_data[["Open", "High", "Low", "Close", "Volume"]]
    df_data = df_data.reset_index()
    df_data.rename(columns={"Open time": "datetime"}, inplace=True)
    df_data["datetime"] = pd.to_datetime(df_data["datetime"])
    df_data = df_data.set_index("datetime")
    return df_data

def pickle_to_df_full(data_path):
    # load the data
    binance_data = vbt.BinanceData.load(data_path)
    df_data = binance_data.get()

    # df_data = df_data[["Open", "High", "Low", "Close", "Volume"]]
    df_data = df_data.reset_index()
    df_data.rename(columns={"Open time": "datetime"}, inplace=True)
    df_data["datetime"] = pd.to_datetime(df_data["datetime"])
    df_data = df_data.set_index("datetime")
    return df_data

def check_min_resolution(df_data):
    df_data = df_data.resample("T").asfreq()

    missing_data = df_data[df_data["Close"].isna()]

    missing_data.to_csv("missing_ccxt.csv")
    return len(missing_data)

def dollar_bar_func(ohlc_df, dollar_bar_size):
    # Calculate dollar value traded for each row
    ohlc_df['DollarValue'] = ohlc_df['Close'] * ohlc_df['Volume']
    
    # Calculate cumulative dollar value
    ohlc_df['CumulativeDollarValue'] = ohlc_df['DollarValue'].cumsum()
    
    # Determine the number of dollar bars
    num_bars = int(ohlc_df['CumulativeDollarValue'].iloc[-1] / dollar_bar_size)
    
    # Generate index positions for dollar bars
    bar_indices = [0]
    cumulative_value = 0
    for i in range(1, len(ohlc_df)):
        cumulative_value += ohlc_df['DollarValue'].iloc[i]
        if cumulative_value >= dollar_bar_size:
            bar_indices.append(i)
            cumulative_value = 0
    
    # Create a new dataframe with dollar bars
    dollar_bars = []
    for i in range(len(bar_indices) - 1):
        start_idx = bar_indices[i]
        end_idx = bar_indices[i + 1]
        
        dollar_bar = {
            'Open': ohlc_df['Open'].iloc[start_idx],
            'High': ohlc_df['High'].iloc[start_idx:end_idx].max(),
            'Low': ohlc_df['Low'].iloc[start_idx:end_idx].min(),
            'Close': ohlc_df['Close'].iloc[end_idx-1],
            'Volume': ohlc_df['Volume'].iloc[start_idx:end_idx].sum(),
        #     'Quote volume': ohlc_df['Quote volume'].iloc[start_idx:end_idx].sum(),
        #     'Trade count': ohlc_df['Trade count'].iloc[start_idx:end_idx].sum(),
        #     'Taker base volume': ohlc_df['Taker base volume'].iloc[start_idx:end_idx].sum(),
        #     'Taker quote volume': ohlc_df['Taker quote volume'].iloc[start_idx:end_idx].sum()
        }
        
        #If the index is 
        if isinstance(ohlc_df.index, pd.DatetimeIndex):
            dollar_bar['Open Time'] = ohlc_df.index[start_idx]
            dollar_bar['Close Time'] = ohlc_df.index[end_idx] - pd.Timedelta(milliseconds=1)
        elif 'Open Time' in ohlc_df.columns:
            dollar_bar['Open Time'] = ohlc_df['Open Time'].iloc[start_idx]
            dollar_bar['Close Time'] = ohlc_df['Open Time'].iloc[end_idx] - pd.Timedelta(milliseconds=1)
        
        dollar_bars.append(dollar_bar)
    
    dollar_bars_df = pd.concat([pd.DataFrame([bar]) for bar in dollar_bars], ignore_index=True)
    
    return dollar_bars_df

def calc_acc_stats(trades_df, initial_capital):
    def custom_aggregation(group):
        total_size = group['Size'].sum()
        total_pnl = group['PnL'].sum()
        count = len(group)
        # Sort the group by 'EntryTime' in ascending order
        sorted_group = group.sort_values('EntryTime')
        
        # Select the first row (earliest entry) from the sorted group
        first_entry = sorted_group.iloc[0]
        
        first_entry_price = first_entry['EntryPrice']
        first_entry_size = first_entry['Size']
        first_entry_time  = first_entry['EntryTime']
        first_exit_time  = first_entry['ExitTime']

        
        return pd.Series([first_entry_time, first_exit_time, total_size, first_entry_price, first_entry_size, total_pnl, count], index=['FirstEntryTime', 'ExitTime', 'TotalSize', 'FirstEntryPrice', 'FirstEntrySize', 'TotalPnL', 'RowCount'])

    # Group by 'ExitBar' and aggregate the data using the custom function
    result = trades_df.groupby('ExitBar').apply(custom_aggregation).reset_index()
    # result = trades_df.groupby("ExitBar")["PnL"].agg(["sum", "count"]).reset_index()
    # result = result.rename(columns={"sum": "TotalPnL", "count": "RowCount"})

    result["portfolio"] = 0
    result["adj pnl"] = 0

    capital = initial_capital
    for index, row in result.iterrows():
        result.at[index, "portfolio"] = capital
        new_value = capital + row["TotalPnL"]

        adj_pnl = (new_value / capital) - 1
        result.at[index, "adj pnl"] = adj_pnl
        capital = new_value

    # result.to_csv("trade_analysis_orig.csv")

    positive = (result["adj pnl"] > 0).sum()
    negative = (result["adj pnl"] < 0).sum()

    win_rate = positive / (positive + negative) * 100
    print(f"Win Rate: {win_rate}")

    average_losing_trade = result[result["adj pnl"] < 0]["adj pnl"].mean()
    print(f"Average Losing Trade: {average_losing_trade}")

    average_winning_trade = result[result["adj pnl"] > 0]["adj pnl"].mean()
    print(f"Average Winning Trade: {average_winning_trade}")

    best_trade = result[result["adj pnl"] == result["adj pnl"].max()]["adj pnl"].values[
        0
    ]
    print(f"Best Trade: {best_trade}")

    worst_trade = result[result["adj pnl"] == result["adj pnl"].min()][
        "adj pnl"
    ].values[0]
    print(f"Worst Trade: {worst_trade}")

    percent = 0.04
    worse_than_minus_0_04 = result[result["adj pnl"] < -percent]
    number_of_trades_worse_than_minus_0_04 = len(worse_than_minus_0_04)
    print(
        f"Trades worse than {percent*100}% : {number_of_trades_worse_than_minus_0_04}"
    )

    first_portfolio_value = result.iloc[0]["portfolio"]
    last_portfolio_value = result.iloc[-1]["portfolio"]
    total_return = (last_portfolio_value / first_portfolio_value) * 100

    print(f"Total Return {total_return}%")

    result["cummax"] = result["portfolio"].cummax()

    # Calculate the drawdown as a percentage of the cumulative maximum
    result["drawdown_percent"] = (
        (result["cummax"] - result["portfolio"]) / result["cummax"]
    ) * 100

    # Find the maximum drawdown and its end date
    max_drawdown_percent = result["drawdown_percent"].max()

    print(f"Max Drawdown: {max_drawdown_percent}%")

    # filtering based on rowcount

    rowcounts = [1, 2, 3, 4, 5]
    try:
        for num in rowcounts:
            filtered_result = result[result["RowCount"] == num]

            positive_rowcount = (filtered_result["adj pnl"] > 0).sum()
            negative_rowcount = (filtered_result["adj pnl"] < 0).sum()

            win_rate_rowcount = (
                positive_rowcount / (positive_rowcount + negative_rowcount)
            ) * 100
            average_losing_trade_rowcount = filtered_result[
                filtered_result["adj pnl"] < 0
            ]["adj pnl"].mean()
            average_winning_trade_rowcount = filtered_result[
                filtered_result["adj pnl"] > 0
            ]["adj pnl"].mean()
            best_trade_rowcount = filtered_result[
                filtered_result["adj pnl"] == filtered_result["adj pnl"].max()
            ]["adj pnl"].values[0]
            worst_trade_rowcount = filtered_result[
                filtered_result["adj pnl"] == filtered_result["adj pnl"].min()
            ]["adj pnl"].values[0]

            percent = 0.05
            worse_than_minus_0_04_rowcount = filtered_result[
                filtered_result["adj pnl"] < -percent
            ]
            number_of_trades_worse_than_minus_0_04_rowcount = len(
                worse_than_minus_0_04_rowcount
            )

            expected_value = (
                win_rate_rowcount / 100
            ) * average_winning_trade_rowcount + (
                1 - win_rate_rowcount / 100
            ) * average_losing_trade_rowcount

            print()
            print(f"RowCount = {num}:")
            print(f"Positive Trades: {positive_rowcount}")
            print(f"Negative Trades: {negative_rowcount}")
            print(f"Win Rate: {win_rate_rowcount:.2f}%")
            print(f"Average Losing Trade: {average_losing_trade_rowcount:.6f}")
            print(f"Average Winning Trade: {average_winning_trade_rowcount:.6f}")
            print(f"Best Trade: {best_trade_rowcount:.6f}")
            print(f"Worst Trade: {worst_trade_rowcount:.6f}")
            print(
                f"Trades worse than {percent*100}%: {number_of_trades_worse_than_minus_0_04_rowcount}"
            )
            print(f"Expected Value of Trade: {expected_value}")
    except:
        pass

    return result
