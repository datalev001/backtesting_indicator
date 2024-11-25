import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime

stock_df = pd.read_csv(r'C:\backupcgi\final_bak\stock_fullsegment.csv')
stock_df.columns

#424
len(set(stock_df.Ticker))

max(stock_df['Date'])
min(stock_df['Date'])

stock_df.dtypes

Date is object
'2019-10-01 00:00:00-04:00'
'2024-10-18 00:00:00-04:00'


##############

def precompute_macd(stock_df, fast=12, slow=26, signal=9):
    """
    Precompute MACD values for all stocks in the dataset.
    """
    stock_df['12-day EMA'] = stock_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=fast).mean())
    stock_df['26-day EMA'] = stock_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=slow).mean())
    stock_df['MACD'] = stock_df['12-day EMA'] - stock_df['26-day EMA']
    stock_df['Signal Line'] = stock_df.groupby('Ticker')['MACD'].transform(lambda x: x.ewm(span=signal).mean())
    return stock_df

def backtest_buyandhold(file_path, start_test_date, end_test_date, initial_capital=10000):
    
    stock_df = pd.read_csv(file_path)
       
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)
    start_test_date = pd.Timestamp(start_test_date, tz='UTC')
    end_test_date = pd.Timestamp(end_test_date, tz='UTC')
    stock_df = stock_df[(stock_df['Date'] >= start_test_date) & (stock_df['Date'] <= end_test_date)]
    stock_df.sort_values(by=['Ticker', 'Date'], inplace=True)

    results = []
    tickers = stock_df['Ticker'].unique()
    
    stock_df = precompute_macd(stock_df)

    # Strategy 1: Individual Buy and Hold
    individual_buy_hold_gains = []
    individual_buy_hold_tickers = []
    for ticker in tickers:
        stock_data = stock_df[stock_df['Ticker'] == ticker]
        if stock_data.empty:
            continue
        buy_price = stock_data.iloc[0]['Open']
        sell_price = stock_data.iloc[-1]['Close']
        gain = (sell_price - buy_price) / buy_price * initial_capital
        individual_buy_hold_gains.append(gain)
        individual_buy_hold_tickers.append(ticker)

    # Strategy 2: QQQ Buy and Hold
    qqq_data = stock_df[stock_df['Ticker'] == 'QQQ']
    qqq_buy_price = qqq_data.iloc[0]['Open'] if not qqq_data.empty else None
    qqq_sell_price = qqq_data.iloc[-1]['Close'] if not qqq_data.empty else None
    qqq_gain = ((qqq_sell_price - qqq_buy_price) / qqq_buy_price * initial_capital) if qqq_buy_price else 0

    # Strategy 3: SPY Buy and Hold
    spy_data = stock_df[stock_df['Ticker'] == 'SPY']
    spy_buy_price = spy_data.iloc[0]['Open'] if not spy_data.empty else None
    spy_sell_price = spy_data.iloc[-1]['Close'] if not spy_data.empty else None
    spy_gain = ((spy_sell_price - spy_buy_price) / spy_buy_price * initial_capital) if spy_buy_price else 0
   
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
            np.mean(individual_buy_hold_gains),
            qqq_gain,
            spy_gain
          ]
        })

    return summary_results

file_path = r'C:\backupcgi\final_bak\stock_fullsegment.csv'
start_test_date = "2019-10-01"
end_test_date = "2024-10-18"
    
backtest_df = backtest_buyandhold(file_path,
                                  start_test_date,
                                  end_test_date, 
                                  initial_capital=10000)

##################updated buy and hold########################

def backtest_buyandhold_up(file_path, start_test_date, end_test_date, initial_capital=10000):
    """
    Backtest Buy and Hold strategy for individual stocks, QQQ, and SPY.

    Args:
        file_path (str): Path to the stock data CSV file.
        start_test_date (str): Start date for the backtest in 'YYYY-MM-DD' format.
        end_test_date (str): End date for the backtest in 'YYYY-MM-DD' format.
        initial_capital (float): Starting capital for the backtest.

    Returns:
        summary_results (DataFrame): Summary of buy-and-hold results.
        percentiles (DataFrame): Percentile returns for individual stock gains.
    """
    stock_df = pd.read_csv(file_path)

    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)
    start_test_date = pd.Timestamp(start_test_date, tz='UTC')
    end_test_date = pd.Timestamp(end_test_date, tz='UTC')
    stock_df = stock_df[(stock_df['Date'] >= start_test_date) & (stock_df['Date'] <= end_test_date)]
    stock_df.sort_values(by=['Ticker', 'Date'], inplace=True)

    tickers = stock_df['Ticker'].unique()

    # Strategy 1: Individual Buy and Hold
    individual_buy_hold_gains = []
    individual_buy_hold_tickers = []
    for ticker in tickers:
        stock_data = stock_df[stock_df['Ticker'] == ticker]
        if stock_data.empty:
            continue
        buy_price = stock_data.iloc[0]['Open']
        sell_price = stock_data.iloc[-1]['Close']
        gain = (sell_price - buy_price) / buy_price * initial_capital
        individual_buy_hold_gains.append(gain)
        individual_buy_hold_tickers.append(ticker)

    # Strategy 2: QQQ Buy and Hold
    qqq_data = stock_df[stock_df['Ticker'] == 'QQQ']
    qqq_buy_price = qqq_data.iloc[0]['Open'] if not qqq_data.empty else None
    qqq_sell_price = qqq_data.iloc[-1]['Close'] if not qqq_data.empty else None
    qqq_gain = ((qqq_sell_price - qqq_buy_price) / qqq_buy_price * initial_capital) if qqq_buy_price else 0

    # Strategy 3: SPY Buy and Hold
    spy_data = stock_df[stock_df['Ticker'] == 'SPY']
    spy_buy_price = spy_data.iloc[0]['Open'] if not spy_data.empty else None
    spy_sell_price = spy_data.iloc[-1]['Close'] if not spy_data.empty else None
    spy_gain = ((spy_sell_price - spy_buy_price) / spy_buy_price * initial_capital) if spy_buy_price else 0

    # Calculate percentiles for individual stock gains
    percentiles = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    percentile_values = np.percentile(individual_buy_hold_gains, percentiles)
    percentiles_df = pd.DataFrame({
        'Percentile': percentiles,
        'Gain': percentile_values
    })

    # Summary results
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
            np.mean(individual_buy_hold_gains),
            qqq_gain,
            spy_gain
        ]
    })

    return summary_results, percentiles_df


# Parameters
file_path = r'C:\backupcgi\final_bak\stock_fullsegment.csv'
start_test_date = "2019-10-01"
end_test_date = "2024-10-18"

# Run backtest
buyhold_results, buyhold_percentiles = backtest_buyandhold_up(
    file_path=file_path,
    start_test_date=start_test_date,
    end_test_date=end_test_date,
    initial_capital=10000
)

# Print results
print(buyhold_results)
print(buyhold_percentiles)

              Strategy    Total Gain  Average Gain/Return
0  Individual Buy and Hold  3.579250e+06          8441.627092
1         QQQ Buy and Hold  1.674480e+04         16744.799094
2         SPY Buy and Hold  1.112902e+04         11129.024245
   Percentile           Gain
0         100  304505.168243
1          90   19385.851774
2          80   11729.980026
3          70    8819.285309
4          60    7431.391232
5          50    5714.785425
6          40    4091.228683
7          30    2728.330338
8          20    1100.487844
9          10   -2514.491824

##################################################


def individual_macd_test(stock_df, start_test_date,
                         end_test_date, initial_capital=10000, 
                         gain_threshold=0.0001, max_holding_days=30, 
                         transaction_cost=0.001, use_end_date=True):
    """
    Perform individual stock MACD testing with a realistic selling rule.

    Selling Rule:
        - If `use_end_date` is False:
            - Sell if MACD < Signal Line (bear market signal) AND gain is above the specified gain_threshold.
            - If no sell condition is met, sell at the end of max_holding_days.
        - If `use_end_date` is True:
            - Hold until the dataset's last date if gain_threshold is not met.

    Args:
        stock_df (DataFrame): DataFrame with stock data.
        start_test_date (str): Start date for testing.
        end_test_date (str): End date for testing.
        initial_capital (float): Starting capital.
        gain_threshold (float): Minimum gain threshold for selling.
        max_holding_days (int): Maximum holding period in days.
        transaction_cost (float): Transaction cost as a percentage.
        use_end_date (bool): Flag to use end date for selling if gain_threshold is not met.

    Returns:
        gains (list): List of gains for each ticker.
        tickers_used (list): List of tickers used in trading.
        transaction_table (DataFrame): Detailed transaction table.
        average_gain (float): Average gain across all tickers.
        confidence_interval (tuple): 95% confidence interval for the gains.
        quantile_table (DataFrame): Quantile return table with specified quantiles and corresponding gains.
    """
    tickers = stock_df['Ticker'].unique()
    transaction_table = []
    gains = []
    tickers_used = []

    for ticker in tickers:
        stock_data = stock_df[stock_df['Ticker'] == ticker]
        stock_data = stock_data[(stock_data['Date'] >= start_test_date) & (stock_data['Date'] <= end_test_date)]
        if stock_data.empty:
            continue

        capital = initial_capital
        for i, row in stock_data.iterrows():
            if row['MACD'] > row['Signal Line']:
                buy_price = row['Close']
                buy_date = row['Date']

                # Determine sell period based on flag
                if use_end_date:
                    adjusted_max_holding_date = stock_data['Date'].max()
                else:
                    adjusted_max_holding_date = min(buy_date + timedelta(days=max_holding_days), stock_data['Date'].max())

                # Find sell candidates within the adjusted holding period
                sell_candidates = stock_data[(stock_data['Date'] > buy_date) & 
                                              (stock_data['Date'] <= adjusted_max_holding_date)]

                # Default to holding until the adjusted maximum holding period ends
                sell_price = sell_candidates['Close'].iloc[-1] if not sell_candidates.empty else buy_price
                sell_date = sell_candidates['Date'].iloc[-1] if not sell_candidates.empty else buy_date

                # Check for MACD < Signal Line and gain above the gain_threshold within sell_candidates
                for _, sell_row in sell_candidates.iterrows():
                    if sell_row['MACD'] < sell_row['Signal Line'] and sell_row['Close'] >= buy_price * (1 + gain_threshold):
                        sell_price = sell_row['Close']
                        sell_date = sell_row['Date']
                        break

                # Apply transaction cost
                gain = (sell_price - buy_price) - (transaction_cost * buy_price + transaction_cost * sell_price)
                capital += gain

                # Record transactions
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

        gain = capital - initial_capital
        gains.append(gain)
        tickers_used.append(ticker)

    # Calculate average gain
    average_gain = np.mean(gains) if gains else 0

    # Calculate 95% confidence interval for gains
    if len(gains) > 1:
        std_dev_gain = np.std(gains, ddof=1)  # Sample standard deviation
        n = len(gains)
        standard_error = std_dev_gain / np.sqrt(n)
        z_score = norm.ppf(0.975)  # 95% confidence interval
        margin_of_error = z_score * standard_error
        confidence_interval = (average_gain - margin_of_error, average_gain + margin_of_error)
    else:
        confidence_interval = (np.nan, np.nan)  # Not enough data for confidence interval

    # Create quantile table
    quantiles = [i for i in range(100, 0, -10)]
    quantile_values = np.percentile(gains, quantiles)
    quantile_table = pd.DataFrame({
        'Quantile': quantiles,
        'Gain': quantile_values
    })

    return gains, tickers_used, pd.DataFrame(transaction_table), average_gain, confidence_interval, quantile_table


def backtest_strategies(stock_df, start_test_date, end_test_date,
                        initial_capital, max_holding_days,
                        gain_threshold, use_end_date, trade_size):
    
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)
    start_test_date = pd.Timestamp(start_test_date, tz='UTC')
    end_test_date = pd.Timestamp(end_test_date, tz='UTC')
    stock_df = stock_df[(stock_df['Date'] >= start_test_date) & (stock_df['Date'] <= end_test_date)]
    stock_df.sort_values(by=['Ticker', 'Date'], inplace=True)

    # Precompute MACD
    stock_df = precompute_macd(stock_df)
  
    transaction_cost = 0.001  # Transaction cost of 0.1%
 
    # Run individual MACD test
    gains, tickers_used, transaction_table, average_gain,\
    confidence_interval, quantile_table = individual_macd_test(\
    stock_df, start_test_date, end_test_date,\
    initial_capital, gain_threshold, max_holding_days,
    transaction_cost, use_end_date)
    
    tot_gains = sum(gains)
    median_gain = np.median(gains) if gains else 0
    selling_rule = 'gain_threshold:' + str(gain_threshold) +\
                  ',max_holding_days:' + str(max_holding_days) +\
                  ',use_end_date=' + str(use_end_date)
    
    # Summary of results
    summary_results = pd.DataFrame({
        'Strategy': [
            'Individual MACD'
        ],
        'Selling_Rule':[
            selling_rule
        ],
        'Median_Gain': [
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

def backtest_multiple_stocks_macd(transaction_table, initial_capital=10000):
    """
    Perform MACD backtest across multiple stocks using extracted transactions.

    Selling Rule:
        - Ensure buy-sell order is valid (cannot buy before selling another stock).
        - Use the sequence of transactions from the individual MACD backtest.

    Args:
        transaction_table (DataFrame): Individual MACD transactions.
        initial_capital (float): Starting capital for the backtest.

    Returns:
        total_gain (float): Total gain from the multiple stock backtest.
        avg_gain (float): Average gain per transaction.
        median_gain (float): Median gain across all trades.
        transaction_log (DataFrame): Log of valid multiple MACD transactions.
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


def execute_macd_backtest(file_path, start_test_date, end_test_date,
                          initial_capital, max_holding_days, 
                          gain_threshold, use_end_date, 
                          trade_size=1000):
    """
    Execute the MACD backtest with realistic trading logic.

    Args:
        file_path (str): Path to the input CSV file containing stock data.
        start_test_date (str): Start date of the backtest in 'YYYY-MM-DD' format.
        end_test_date (str): End date of the backtest in 'YYYY-MM-DD' format.
        initial_capital (float): Starting capital for the backtest.
        max_holding_days (int): Maximum holding period (in days).
        gain_threshold (float): Gain threshold for selling (e.g., 0.001 for 0.1% gain).
        trade_size (float): Minimum capital allocation for each trade.

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
    #stock_df = stock_df[stock_df['Ticker'].isin(sampled_tickers)]

    # Run individual MACD backtest strategy
    summary, transaction_table, quantile_table = backtest_strategies(
        stock_df=stock_df,
        start_test_date=start_test_date,
        end_test_date=end_test_date,
        initial_capital=initial_capital,
        max_holding_days=max_holding_days,
        gain_threshold=gain_threshold,
        use_end_date = use_end_date,
        trade_size=trade_size
    )

    # Run multiple MACD backtest strategy
    total_gain, avg_gain, median_gain, multiple_transaction_log = backtest_multiple_stocks_macd(
        transaction_table=transaction_table,
        initial_capital=initial_capital
    )

    # Create summary for multiple MACD backtest
    selling_rule = 'gain_threshold:' + str(gain_threshold) +\
                  ',max_holding_days:' + str(max_holding_days) +\
                  ',use_end_date=' + str(use_end_date)
    
    multiple_macd_summary = pd.DataFrame({
        'Strategy': ['Multiple Stocks MACD'],
        'Selling_Rule': [selling_rule],
        'Median_Gain': [median_gain],
        'Average_Gain': [avg_gain],
        'Confidence': [None]
    })

    # Concatenate summaries
    summary = pd.concat([summary, multiple_macd_summary], ignore_index=True)
    quantile_table['Selling Rule'] = selling_rule
    return summary, transaction_table, quantile_table

file_path = r'C:\backupcgi\final_bak\stock_fullsegment.csv'
start_test_date = "2019-10-01"
end_test_date = "2024-10-18"
max_holding_days = 100
gain_threshold = 0.05
initial_capital=10000
trade_size = 1000
use_end_date = True

summary, transaction_table, quantile_table =\
  execute_macd_backtest(file_path, start_test_date, 
                            end_test_date,
                            initial_capital, 
                            max_holding_days, 
                            gain_threshold, 
                            use_end_date,
                            trade_size)
 
    
'''
max_holding_days = 100
gain_threshold = 0.05
use_end_date = False
'''

RULES = [(15, 0.01, False), (30, 0.02, False), (45, 0.02, False),
 (60, 0.03, False), (90, 0.03, False), (100, 0.03, False),
 (120, 0.05, True), (150, 0.05, True), (200, 0.05, True)]

summary_df = pd.DataFrame([])
quantile_df = pd.DataFrame([])

for itv in RULES:
    max_holding_days = itv[0]
    gain_threshold = itv[1]
    use_end_date = itv[2]

    summary, transaction_table, quantile_table =\
      execute_macd_backtest(file_path, start_test_date, 
                                end_test_date, initial_capital, 
                                max_holding_days, gain_threshold, 
                                use_end_date, trade_size)
  
    summary_df = pd.concat([summary_df,summary])
    quantile_df = pd.concat([quantile_df,quantile_table])


summary_df.to_csv(r'C:\backupcgi\final_bak\summary_MACD.csv',index = False)
quantile_df.to_csv(r'C:\backupcgi\final_bak\quantile_MACD.csv',index = False)
backtest_df.to_csv(r'C:\backupcgi\final_bak\buyhold_MACD.csv',index = False)









import matplotlib.pyplot as plt
import pandas as pd

# Example data
data = {
    'Day': ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', 'Current'],
    'MACD_Line': [2.00, 1.80, 1.70, 2.10, 2.20, 2.50, 2.40, 2.60, 2.70, 2.00],  # MACD values
    'Signal_Line': [None, None, None, None, None, None, None, None, None, 2.30],  # 9-day EMA of MACD Line
    'MACD_Histogram': [None, None, None, None, None, None, None, None, None, -0.30],  # Difference between MACD and Signal Line
}

# Create DataFrame
df = pd.DataFrame(data)

# Plot MACD Line and Signal Line
plt.figure(figsize=(10, 6))
plt.plot(df['Day'], df['MACD_Line'], label='MACD Line', color='blue', marker='o')
plt.plot(df['Day'], df['Signal_Line'], label='Signal Line', color='orange', marker='o')

# Add MACD Histogram as bars
histogram_values = [val if val is not None else 0 for val in df['MACD_Histogram']]  # Replace None with 0
plt.bar(df['Day'], histogram_values, label='MACD Histogram', color='gray', alpha=0.5)

# Add horizontal line at 0 for reference
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)

# Add labels, legend, and title
plt.title('MACD Indicator Chart')
plt.xlabel('Day')
plt.ylabel('Value')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
