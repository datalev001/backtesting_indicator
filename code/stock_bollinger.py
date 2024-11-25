import pandas as pd
import numpy as np
from datetime import timedelta
from random import sample
from scipy.stats import norm


def precompute_bollinger_bands(stock_df, period=20, std_dev=2):
    """
    Precompute Bollinger Bands for all stocks in the dataset.
    """
    def calculate_bollinger(series):
        rolling_mean = series.rolling(window=period).mean()
        rolling_std = series.rolling(window=period).std()
        upper_band = rolling_mean + (std_dev * rolling_std)
        lower_band = rolling_mean - (std_dev * rolling_std)
        return rolling_mean, upper_band, lower_band

    stock_df['SMA'] = stock_df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=period).mean()
    )
    stock_df['Upper Band'] = stock_df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=period).mean() + std_dev * x.rolling(window=period).std()
    )
    stock_df['Lower Band'] = stock_df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=period).mean() - std_dev * x.rolling(window=period).std()
    )
    return stock_df


def backtest_buyandhold(file_path, start_test_date, end_test_date, initial_capital=10000):
    """
    Backtest buy-and-hold strategy for individual stocks, QQQ, and SPY.
    """
    stock_df = pd.read_csv(file_path)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)
    start_test_date = pd.Timestamp(start_test_date, tz='UTC')
    end_test_date = pd.Timestamp(end_test_date, tz='UTC')
    stock_df = stock_df[(stock_df['Date'] >= start_test_date) & (stock_df['Date'] <= end_test_date)]
    stock_df.sort_values(by=['Ticker', 'Date'], inplace=True)

    tickers = stock_df['Ticker'].unique()

    individual_buy_hold_gains = []
    for ticker in tickers:
        stock_data = stock_df[stock_df['Ticker'] == ticker]
        if stock_data.empty:
            continue
        buy_price = stock_data.iloc[0]['Open']
        sell_price = stock_data.iloc[-1]['Close']
        gain = (sell_price - buy_price) / buy_price * initial_capital
        individual_buy_hold_gains.append(gain)

    # QQQ Buy and Hold
    qqq_data = stock_df[stock_df['Ticker'] == 'QQQ']
    qqq_gain = 0
    if not qqq_data.empty:
        qqq_buy_price = qqq_data.iloc[0]['Open']
        qqq_sell_price = qqq_data.iloc[-1]['Close']
        qqq_gain = ((qqq_sell_price - qqq_buy_price) / qqq_buy_price) * initial_capital

    # SPY Buy and Hold
    spy_data = stock_df[stock_df['Ticker'] == 'SPY']
    spy_gain = 0
    if not spy_data.empty:
        spy_buy_price = spy_data.iloc[0]['Open']
        spy_sell_price = spy_data.iloc[-1]['Close']
        spy_gain = ((spy_sell_price - spy_buy_price) / spy_buy_price) * initial_capital

    summary_results = pd.DataFrame({
        'Strategy': [
            'Individual Buy and Hold',
            'QQQ Buy and Hold',
            'SPY Buy and Hold'
        ],
        'Total Gain': [
            sum(individual_buy_hold_gains),
            qqq_gain,
            spy_gain
        ],
        'Average Gain/Return': [
            np.mean(individual_buy_hold_gains) if individual_buy_hold_gains else 0,
            qqq_gain,
            spy_gain
        ]
    })

    return summary_results


def individual_bollinger_test(stock_df, start_test_date, end_test_date,  
                              initial_capital=10000, gain_threshold=0.01, 
                              max_holding_days=30, transaction_cost=0.001, 
                              use_end_date=False, delta_low=0.0, delta_up=0.0):
    """
    Perform individual stock Bollinger Bands testing with a realistic selling rule.

    Selling Rule:
        - Buy when Close < Lower Band * (1 - delta_low).
        - Sell when Close > Upper Band * (1 + delta_up) AND gain is above the gain_threshold.
        - If `use_end_date` is True, hold until the last dataset date.
        - Otherwise, sell at the end of max_holding_days if gain_threshold is not met.

    Args:
        stock_df (DataFrame): Stock data.
        start_test_date (str): Start date for testing.
        end_test_date (str): End date for testing.
        initial_capital (float): Starting capital.
        gain_threshold (float): Minimum gain threshold for selling.
        max_holding_days (int): Maximum holding period in days.
        transaction_cost (float): Transaction cost as a percentage.
        use_end_date (bool): Flag to use the end date of the data for selling.
        delta_low (float): Adjustment parameter for the lower band in buying rule (percentage).
        delta_up (float): Adjustment parameter for the upper band in selling rule (percentage).

    Returns:
        gains (list): List of gains for each ticker.
        transaction_table (DataFrame): Detailed transaction table.
        average_gain (float): Average gain across all tickers.
        confidence_interval (tuple): 95% confidence interval for the gains.
        quantile_table (DataFrame): Quantile return table with specified quantiles and corresponding gains.
    """
       
    tickers = stock_df['Ticker'].unique()
    transaction_table = []
    gains = []

    for ticker in tickers:
        stock_data = stock_df[stock_df['Ticker'] == ticker]
        stock_data = stock_data[(stock_data['Date'] >= start_test_date) & (stock_data['Date'] <= end_test_date)]
        if stock_data.empty:
            continue

        for i, row in stock_data.iterrows():
            # Adjust the buying rule with delta_low as a percentage
            if row['Close'] < row['Lower Band'] * (1 - delta_low):
                buy_price = row['Close']
                buy_date = row['Date']

                adjusted_max_holding_date = stock_data['Date'].max() if use_end_date else \
                    min(buy_date + timedelta(days=max_holding_days), stock_data['Date'].max())

                sell_candidates = stock_data[(stock_data['Date'] > buy_date) & 
                                              (stock_data['Date'] <= adjusted_max_holding_date)]

                sell_price = sell_candidates['Close'].iloc[-1] if not sell_candidates.empty else buy_price
                sell_date = sell_candidates['Date'].iloc[-1] if not sell_candidates.empty else buy_date

                for _, sell_row in sell_candidates.iterrows():
                    # Adjust the selling rule with delta_up as a percentage
                    if (sell_row['Close'] > sell_row['Upper Band'] * (1 + delta_up) and 
                        sell_row['Close'] >= buy_price * (1 + gain_threshold)):
                        sell_price = sell_row['Close']
                        sell_date = sell_row['Date']
                        break

                gain = (sell_price - buy_price) - (transaction_cost * buy_price + transaction_cost * sell_price)
                gains.append(gain)

                transaction_table.append({
                    'Ticker': ticker,
                    'Action': 'Buy',
                    'Date': buy_date,
                    'Price': buy_price
                })
                transaction_table.append({
                    'Ticker': ticker,
                    'Action': 'Sell',
                    'Date': sell_date,
                    'Price': sell_price
                })

    average_gain = np.mean(gains) if gains else 0

    if len(gains) > 1:
        std_dev_gain = np.std(gains, ddof=1)
        n = len(gains)
        standard_error = std_dev_gain / np.sqrt(n)
        z_score = norm.ppf(0.975)
        margin_of_error = z_score * standard_error
        confidence_interval = (average_gain - margin_of_error, average_gain + margin_of_error)
    else:
        confidence_interval = (np.nan, np.nan)

    quantiles = [i for i in range(100, 0, -10)]
    quantile_values = np.percentile(gains, quantiles) if gains else []
    quantile_table = pd.DataFrame({'Quantile': quantiles, 'Gain': quantile_values})

    return gains, pd.DataFrame(transaction_table), average_gain, confidence_interval, quantile_table


def backtest_strategies_bollinger(stock_df, start_test_date, end_test_date, 
                                  initial_capital, max_holding_days,
                                  gain_threshold, use_end_date, trade_size,
                                  delta_low=0.0, delta_up=0.0):
    """
    Backtest strategies using Bollinger Bands-based trading rules.

    Args:
        stock_df (DataFrame): Stock data.
        start_test_date (str): Start date for backtesting.
        end_test_date (str): End date for backtesting.
        initial_capital (float): Starting capital for the backtest.
        max_holding_days (int): Maximum holding period.
        gain_threshold (float): Minimum gain required for selling.
        use_end_date (bool): Whether to use the end date of data for selling.
        trade_size (float): Trade size (not used in Bollinger Band logic but kept for consistency).
        delta_low (float): Adjustment for lower band in buying rule.
        delta_up (float): Adjustment for gain threshold in selling rule.

    Returns:
        summary_results (DataFrame): Summary of backtest results.
        transaction_table (DataFrame): Detailed log of transactions.
        quantile_table (DataFrame): Quantile return table.
    """
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)
    start_test_date = pd.Timestamp(start_test_date, tz='UTC')
    end_test_date = pd.Timestamp(end_test_date, tz='UTC')
    stock_df = stock_df[(stock_df['Date'] >= start_test_date) & (stock_df['Date'] <= end_test_date)]
    stock_df.sort_values(by=['Ticker', 'Date'], inplace=True)

    # Precompute Bollinger Bands
    stock_df = precompute_bollinger_bands(stock_df)  # Ensure you have this function implemented

    transaction_cost = 0.001  # Transaction cost of 0.1%

    # Run individual Bollinger Bands test
    gains, transaction_table, average_gain, confidence_interval, quantile_table = individual_bollinger_test(
        stock_df, start_test_date, end_test_date,
        initial_capital, gain_threshold, max_holding_days,
        transaction_cost, use_end_date, delta_low, delta_up
    )

    tot_gains = sum(gains)
    median_gain = np.median(gains) if gains else 0
    selling_rule = f'gain_threshold={gain_threshold}, max_holding_days={max_holding_days}, delta_low={delta_low}, delta_up={delta_up}'

    # Summary of results
    summary_results = pd.DataFrame({
        'Strategy': [
            'Individual Bollinger Bands'
        ],
        'Selling_rule': [
            selling_rule
        ],
        'Median Gain': [
            median_gain,
        ],
        'Average_Gain': [
            average_gain,
        ],
        'Confidence': [
            str(confidence_interval)
        ]
    })

    return summary_results, transaction_table, quantile_table


def backtest_multiple_stocks_bollinger(transaction_table, initial_capital=10000):
    """
    Perform Bollinger Bands backtest across multiple stocks using extracted transactions.

    Selling Rule:
        - Ensure buy-sell order is valid (cannot buy before selling another stock).
        - Use the sequence of transactions from the individual Bollinger Bands backtest.

    Args:
        transaction_table (DataFrame): Individual Bollinger Bands transactions.
        initial_capital (float): Starting capital for the backtest.

    Returns:
        total_gain (float): Total gain from the multiple stock backtest.
        avg_gain (float): Average gain per transaction.
        median_gain (float): Median gain across all trades.
        transaction_log (DataFrame): Log of valid multiple Bollinger Bands transactions.
    """
    # Ensure consistent sorting
    transaction_table['Date'] = pd.to_datetime(transaction_table['Date'], utc=True)
    transaction_table = transaction_table.sort_values(by=['Date', 'Ticker', 'Action']).reset_index(drop=True)

    # Initialize variables
    capital = initial_capital
    total_gain = 0
    transaction_log = []  # Log of all executed transactions
    last_action = None    # Track the last action (Buy/Sell)
    current_ticker = None # Track the current ticker being traded

    for i, row in transaction_table.iterrows():
        if row['Action'] == 'Buy' and (last_action is None or last_action == 'Sell'):
            # Start a new trade
            buy_price = row['Price']
            buy_date = row['Date']
            current_ticker = row['Ticker']
            last_action = 'Buy'

            # Log the buy action
            transaction_log.append({
                'Ticker': row['Ticker'],
                'Action': 'Buy',
                'Date': row['Date'],
                'Price': buy_price,
                'Capital': capital
            })

        elif row['Action'] == 'Sell' and last_action == 'Buy' and row['Ticker'] == current_ticker:
            # Complete the trade
            sell_price = row['Price']
            sell_date = row['Date']

            # Calculate gain
            gain = sell_price - buy_price
            capital += gain
            total_gain += gain

            # Log the sell action
            transaction_log.append({
                'Ticker': row['Ticker'],
                'Action': 'Sell',
                'Date': sell_date,
                'Price': sell_price,
                'Gain': gain,
                'Capital': capital
            })

            # Update state
            last_action = 'Sell'
            current_ticker = None

    # Convert transaction log to DataFrame
    transaction_log_df = pd.DataFrame(transaction_log)

    # Calculate average and median gains
    gains = transaction_log_df[transaction_log_df['Action'] == 'Sell']['Gain']
    avg_gain = gains.mean() if not gains.empty else 0
    median_gain = gains.median() if not gains.empty else 0

    return total_gain, avg_gain, median_gain, transaction_log_df


def execute_bollinger_backtest(file_path, start_test_date, end_test_date, 
                               initial_capital, max_holding_days, 
                               gain_threshold, use_end_date, 
                               trade_size=1000, delta_low=0.0, delta_up=0.0):
    """
    Execute the Bollinger Bands backtest with realistic trading logic.

    Args:
        file_path (str): Path to the input CSV file containing stock data.
        start_test_date (str): Start date of the backtest in 'YYYY-MM-DD' format.
        end_test_date (str): End date of the backtest in 'YYYY-MM-DD' format.
        initial_capital (float): Starting capital for the backtest.
        max_holding_days (int): Maximum holding period (in days).
        gain_threshold (float): Gain threshold for selling (e.g., 0.001 for 0.1% gain).
        trade_size (float): Minimum capital allocation for each trade.
        use_end_date (bool): Whether to use the end date of data for selling.
        delta_low (float): Additional threshold for the lower Bollinger Band for buying.
        delta_up (float): Additional threshold for the upper Bollinger Band for selling.

    Returns:
        summary (DataFrame): Summary of backtest results for all strategies.
        transaction_table (DataFrame): Detailed log of all transactions.
        quantile_table (DataFrame): Quantile return table.
    """
    # Load stock data
    stock_df = pd.read_csv(file_path)

    # Sample 50 random tickers from the dataset
    tickers = list(set(stock_df['Ticker']))
    sampled_tickers = sample(tickers, 50)
    # stock_df = stock_df[stock_df['Ticker'].isin(sampled_tickers)]

    # Run individual Bollinger Bands backtest strategy
    summary, transaction_table, quantile_table = backtest_strategies_bollinger(
        stock_df=stock_df,
        start_test_date=start_test_date,
        end_test_date=end_test_date,
        initial_capital=initial_capital,
        max_holding_days=max_holding_days,
        gain_threshold=gain_threshold,
        use_end_date=use_end_date,
        trade_size=trade_size,
        delta_low=delta_low,
        delta_up=delta_up
    )

    # Run multiple Bollinger Bands backtest strategy
    total_gain, avg_gain, median_gain, multiple_transaction_log = backtest_multiple_stocks_bollinger(
        transaction_table=transaction_table,
        initial_capital=initial_capital
    )

    # Create summary for multiple Bollinger Bands backtest
    selling_rule = f'gain_threshold={gain_threshold}, max_holding_days={max_holding_days}, delta_low={delta_low}, delta_up={delta_up}'
    multiple_bollinger_summary = pd.DataFrame({
        'Strategy': ['Multiple Stocks Bollinger Bands'],
        'Selling Rule': [selling_rule],
        'Median Gain': [median_gain],
        'Average Gain': [avg_gain],
        'Confidence': [None]
    })

    # Concatenate summaries
    summary = pd.concat([summary, multiple_bollinger_summary], ignore_index=True)
    quantile_table['Selling Rule'] = selling_rule
    return summary, transaction_table, quantile_table

# Backtest execution
file_path = r'C:\backupcgi\final_bak\stock_fullsegment.csv'
start_test_date = "2019-10-01"
end_test_date = "2024-10-18"
max_holding_days = 50
gain_threshold = 0.0
initial_capital = 10000
trade_size = 1000
use_end_date = False
delta_low = 0.0  # Additional threshold for Lower Band (buying rule)
delta_up = 0.0   # Additional threshold for Upper Band (selling rule)

summary, transaction_table, quantile_table = execute_bollinger_backtest(
    file_path=file_path,
    start_test_date=start_test_date,
    end_test_date=end_test_date,
    initial_capital=initial_capital,
    max_holding_days=max_holding_days,
    gain_threshold=gain_threshold,
    use_end_date=use_end_date,
    trade_size=trade_size,
    delta_low=delta_low,
    delta_up=delta_up
)

# Output the results
print(summary)
print(quantile_table)



file_path = 'stock_fullsegment.csv'
start_test_date = "2019-10-01"
end_test_date = "2024-10-18"
initial_capital = 10000
trade_size = 1000

max_holding_days = 100
gain_threshold = 0.0
use_end_date = False

R = [0.03]
RULES = [0.01*n for n in range(7,12)]

summary_df = pd.DataFrame([])
quantile_df = pd.DataFrame([])

for itv1 in R:
    delta_low = itv1
    for itv2 in RULES:
        delta_up = itv2    
 
        summary, transaction_table, quantile_table = execute_bollinger_backtest(
            file_path=file_path,
            start_test_date=start_test_date,
            end_test_date=end_test_date,
            initial_capital=initial_capital,
            max_holding_days=max_holding_days,
            gain_threshold=gain_threshold,
            use_end_date=use_end_date,
            trade_size=trade_size,
            delta_low=delta_low,
            delta_up=delta_up
        )
         
        R = [delta_low, delta_up] 
        print (R)
        print ("**************************************")
        print (quantile_table)
        quantile_table_chosen = quantile_table.head(3)
        quantile_df = pd.concat([quantile_df, quantile_table_chosen], ignore_index=True)
        
quantile_df.to_csv('QuantileTable_Bolinger.csv', index = False)    
         

