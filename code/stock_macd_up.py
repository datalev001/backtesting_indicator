import numpy as np
import pandas as pd
from datetime import timedelta

def precompute_macd(file_path, start_test_date, end_test_date, fast=12, slow=26,
                    signal=9, gain_threshold=0.03, max_holding_days=100, 
                    use_end_date=True):
  
    # Load stock data
    stock_df = pd.read_csv(file_path)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)
    
    # Filter dates
    start_test_date = pd.Timestamp(start_test_date, tz='UTC')
    end_test_date = pd.Timestamp(end_test_date, tz='UTC')
    stock_df = stock_df[(stock_df['Date'] >= start_test_date) & (stock_df['Date'] <= end_test_date)]
    stock_df.sort_values(by=['Ticker', 'Date'], inplace=True)
    
    # Calculate MACD and Signal Line
    stock_df['12-day EMA'] = stock_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=fast).mean())
    stock_df['26-day EMA'] = stock_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=slow).mean())
    stock_df['MACD'] = stock_df['12-day EMA'] - stock_df['26-day EMA']
    stock_df['Signal Line'] = stock_df.groupby('Ticker')['MACD'].transform(lambda x: x.ewm(span=signal).mean())

    # Initialize macd_df to record trading points
    macd_records = []

    # Group by Ticker to process each stock individually
    for ticker, group in stock_df.groupby('Ticker'):
        group = group.sort_values('Date')
        holding = False
        buy_price, buy_date = None, None

        for i, row in group.iterrows():
            if not holding and row['MACD'] > row['Signal Line']:
                # Buy signal
                buy_price = row['Close']
                buy_date = row['Date']
                holding = True
                continue

            if holding:
                # Determine max holding period
                max_holding_date = (
                    group['Date'].max() if use_end_date
                    else min(buy_date + timedelta(days=max_holding_days), group['Date'].max())
                )

                # Filter sell candidates
                sell_candidates = group[(group['Date'] > buy_date) & (group['Date'] <= max_holding_date)]

                if not sell_candidates.empty:
                    # Default to holding until the last candidate
                    sell_row = sell_candidates.iloc[-1]
                    sell_price = sell_row['Close']
                    sell_date = sell_row['Date']

                    # Check for MACD < Signal Line and gain threshold
                    for _, candidate in sell_candidates.iterrows():
                        if candidate['MACD'] < candidate['Signal Line'] and candidate['Close'] >= buy_price * (1 + gain_threshold):
                            sell_price = candidate['Close']
                            sell_date = candidate['Date']
                            break

                    # Record buy and sell points
                    macd_records.append({
                        'Ticker': ticker,
                        'Buy Date': buy_date,
                        'Buy Price': buy_price,
                        'Sell Date': sell_date,
                        'Sell Price': sell_price
                    })
                    holding = False

        # Force a sell at the end date if still holding
        if holding:
            sell_price = group['Close'].iloc[-1]
            sell_date = group['Date'].iloc[-1]
            macd_records.append({
                'Ticker': ticker,
                'Buy Date': buy_date,
                'Buy Price': buy_price,
                'Sell Date': sell_date,
                'Sell Price': sell_price
            })
            holding = False

    # Create macd_df from the recorded points
    macd_df = pd.DataFrame(macd_records)

    return stock_df, macd_df

def individual_macd_test_from_macd_df(macd_df,
    transaction_cost=0.001, initial_capital=10000):

    gains = []
    tickers_used = []
    transaction_table = []

    for ticker, group in macd_df.groupby('Ticker'):
        capital = initial_capital
        ticker_gain = 0

        for _, row in group.iterrows():
            # Calculate gain for each trade
            buy_price = row['Buy Price']
            sell_price = row['Sell Price']
            gain = (sell_price - buy_price) - (transaction_cost * (buy_price + sell_price))
            ticker_gain += gain

            # Record transactions
            transaction_table.append({
                'Ticker': ticker,
                'Action': 'Buy',
                'Date': row['Buy Date'],
                'Price': buy_price
            })
            transaction_table.append({
                'Ticker': ticker,
                'Action': 'Sell',
                'Date': row['Sell Date'],
                'Price': sell_price
            })

        gains.append(ticker_gain)
        tickers_used.append(ticker)

    # Calculate average gain
    average_gain = np.mean(gains) if gains else 0

    # Calculate confidence interval
    if len(gains) > 1:
        std_dev_gain = np.std(gains, ddof=1)
        margin_of_error = norm.ppf(0.975) * (std_dev_gain / np.sqrt(len(gains)))
        confidence_interval = (average_gain - margin_of_error, average_gain + margin_of_error)
    else:
        confidence_interval = (np.nan, np.nan)

    # Create quantile table
    quantiles = np.arange(100, 0, -10)
    quantile_values = np.percentile(gains, quantiles) if gains else []
    quantile_table = pd.DataFrame({'Quantile': quantiles, 'Gain': quantile_values})

    return gains, tickers_used, pd.DataFrame(transaction_table), average_gain, confidence_interval, quantile_table


# Usage Example
file_path = 'stock_fullsegment.csv'
start_test_date = "2019-10-01"
end_test_date = "2024-10-18"

# Precompute MACD and extract trading points
stock_df, macd_df = precompute_macd(file_path, start_test_date, end_test_date,
gain_threshold=0.06, max_holding_days=120, use_end_date=True)

# Perform individual MACD test
results = individual_macd_test_from_macd_df(macd_df)
len(results)

# Display results
print("\nSummary:")
print(f"Average Gain: {results[3]}, Confidence Interval: {results[4]}")

print("\nQuantile Table:")
print(results[5])

################################
use the following to test:

1) gain_threshold=0.01, max_holding_days=30, use_end_date=True
2) gain_threshold=0.03, max_holding_days=60, use_end_date=True    
3) gain_threshold=0.05, max_holding_days=80, use_end_date=True
4) gain_threshold=0.06, max_holding_days=100, use_end_date=True
5) gain_threshold=0.09, max_holding_days=120, use_end_date=True    
6) gain_threshold=0.1,  max_holding_days=150, use_end_date=True        
7) gain_threshold=0.01, max_holding_days=30, use_end_date=False
8) gain_threshold=0.03, max_holding_days=60, use_end_date=False
9) gain_threshold=0.05, max_holding_days=80, use_end_date=False
10) gain_threshold=0.06, max_holding_days=100, use_end_date=False
11) gain_threshold=0.09, max_holding_days=120, use_end_date=False    
12) gain_threshold=0.1,  max_holding_days=150, use_end_date=False        

    
1) create QuantileTable_all_df, put strategy on one column,
and quantile j (j = 100, 90,..) gain on onther columns  

2) create Summary_all_df, concate rows of all columns in Summary, 
    Confidence Interval and Average Gain 


scenarios = [
    (0.01, 30, True), (0.03, 60, True), (0.05, 80, True), (0.06, 100, True),
    (0.09, 120, True), (0.1, 150, True),
    (0.01, 30, False), (0.03, 60, False), (0.05, 80, False), (0.06, 100, False),
    (0.09, 120, False), (0.1, 150, False)
]

quantile_table_all = []
summary_all = []

for gain_threshold, max_holding_days, use_end_date in scenarios:
    stock_df, macd_df = precompute_macd(
        file_path, start_test_date, end_test_date,
        gain_threshold=gain_threshold, max_holding_days=max_holding_days, use_end_date=use_end_date
    )
    results = individual_macd_test_from_macd_df(macd_df)
    strategy = f"gain_threshold={gain_threshold}, max_holding_days={max_holding_days}, use_end_date={use_end_date}"

    # Add to quantile table
    quantile_table = results[5].copy()
    quantile_table['Strategy'] = strategy
    quantile_table = quantile_table.pivot_table(index='Strategy', columns='Quantile', values='Gain').reset_index()
    quantile_table_all.append(quantile_table)

    # Add to summary
    summary_all.append({
        'Strategy': strategy,
        'Average Gain': results[3],
        'Median Gain': np.median(results[0]) if results[0] else 0,
        'Confidence': results[4]
    })

# Combine results into DataFrames
QuantileTable_all_df = pd.concat(quantile_table_all, ignore_index=True)
Summary_all_df = pd.DataFrame(summary_all)

# Display results
print("\nQuantile Table:")
print(QuantileTable_all_df)

print("\nSummary Table:")
print(Summary_all_df)


########multiple macd################
def backtest_multiple_stocks_macd(macd_df, initial_capital=10000):
  
    # Ensure consistent sorting
    macd_df['Buy Date'] = pd.to_datetime(macd_df['Buy Date'], utc=True)
    macd_df['Sell Date'] = pd.to_datetime(macd_df['Sell Date'], utc=True)
    macd_df = macd_df.sort_values(by=['Buy Date', 'Ticker']).reset_index(drop=True)

    # Initialize variables
    capital = initial_capital
    total_gain = 0
    transaction_log = []  # Log of all executed transactions
    last_action = None    # Track the last action (Buy/Sell)
    current_ticker = None # Track the current ticker being traded

    for _, row in macd_df.iterrows():
        # Start a new trade (Buy action)
        if last_action is None or last_action == 'Sell':
            buy_price = row['Buy Price']
            buy_date = row['Buy Date']
            current_ticker = row['Ticker']
            last_action = 'Buy'

            # Log the buy action
            transaction_log.append({
                'Ticker': row['Ticker'],
                'Action': 'Buy',
                'Date': buy_date,
                'Price': buy_price,
                'Capital': capital
            })

        # Complete the trade (Sell action)
        if last_action == 'Buy' and row['Ticker'] == current_ticker:
            sell_price = row['Sell Price']
            sell_date = row['Sell Date']

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

file_path = r'C:\backupcgi\final_bak\stock_fullsegment.csv'
start_test_date = "2019-10-01"
end_test_date = "2024-10-18"

# Precompute MACD and extract trading points
stock_df, macd_df = precompute_macd(file_path, start_test_date, end_test_date, gain_threshold=0.05, max_holding_days=100)

# Perform multiple stocks MACD backtest
total_gain, avg_gain, median_gain, transaction_log = backtest_multiple_stocks_macd(macd_df)

# Display results
print(f"Total Gain: {total_gain}")
print(f"Average Gain: {avg_gain}")
print(f"Median Gain: {median_gain}")
print("\nTransaction Log:")
print(transaction_log)

print(f"Median Gain: {median_gain}")
Median Gain: 3.9832592010498047

print(f"Average Gain: {avg_gain}")
Average Gain: -4.955285924529415
###############################################
