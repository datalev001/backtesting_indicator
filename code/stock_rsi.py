import pandas as pd
import numpy as np
from datetime import timedelta
from random import sample
from scipy.stats import norm

def precompute_rsi(stock_df, period=14):
    """
    Precompute RSI values for all stocks in the dataset.
    """
    def calculate_rsi(series):
        delta = series.diff(1)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    stock_df['RSI'] = stock_df.groupby('Ticker')['Close'].transform(calculate_rsi)
    return stock_df


from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.stats import norm

def individual_rsi_test(stock_df, start_test_date, end_test_date, 
                        initial_capital=10000, lags_buy_days=1, gain_threshold=0.02, 
                        max_holding_days=30, transaction_cost=0.001, 
                        use_end_date=False, rsi_entry=30, rsi_exit=70):
    """
    Perform individual stock RSI testing with a realistic buying and selling rule.

    Args:
        stock_df (DataFrame): DataFrame with stock data.
        start_test_date (str): Start date for testing.
        end_test_date (str): End date for testing.
        initial_capital (float): Starting capital.
        lags_buy_days (int): Number of days to wait after RSI threshold for buying.
        gain_threshold (float): Minimum percentage gain for selling.
        max_holding_days (int): Maximum holding period in days.
        transaction_cost (float): Transaction cost as a percentage.
        use_end_date (bool): Flag to use end date for selling if gain_threshold is not met.
        rsi_entry (int): RSI threshold for buying (e.g., 30).
        rsi_exit (int): RSI threshold for selling (e.g., 70).

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
            # Check for RSI buy condition
            if row['RSI'] < rsi_entry:
                # Ensure the buy date index is valid
                buy_date_index = min(stock_data.index.get_loc(i) + lags_buy_days, len(stock_data) - 1)
                buy_date = stock_data.iloc[buy_date_index]['Date']
                buy_price = stock_data.iloc[buy_date_index]['Close']

                # Relaxed price condition
                if buy_price >= row['Close']:  # Allow flat or slightly increasing prices
                    # Determine sell period based on flag
                    if use_end_date:
                        adjusted_max_holding_date = stock_data['Date'].max()
                    else:
                        adjusted_max_holding_date = min(
                            buy_date + timedelta(days=max_holding_days),
                            stock_data['Date'].max()
                        )

                    # Find sell candidates within the adjusted holding period
                    sell_candidates = stock_data[(stock_data['Date'] > buy_date) & 
                                                  (stock_data['Date'] <= adjusted_max_holding_date)]

                    # Default sell price and date
                    sell_price = sell_candidates['Close'].iloc[-1] if not sell_candidates.empty else buy_price
                    sell_date = sell_candidates['Date'].iloc[-1] if not sell_candidates.empty else buy_date

                    # Check for RSI sell condition
                    for _, sell_row in sell_candidates.iterrows():
                        if (sell_row['RSI'] > rsi_exit) and \
                           (sell_row['Close'] >= buy_price * (1 + gain_threshold)):
                            sell_price = sell_row['Close']
                            sell_date = sell_row['Date']
                            break

                    # Calculate gain and update capital
                    gain = (sell_price - buy_price) - (transaction_cost * buy_price + transaction_cost * sell_price)
                    capital += gain
                    gains.append(gain)

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
        std_dev_gain = np.std(gains, ddof=1)
        n = len(gains)
        standard_error = std_dev_gain / np.sqrt(n)
        z_score = norm.ppf(0.975)
        margin_of_error = z_score * standard_error
        confidence_interval = (average_gain - margin_of_error, average_gain + margin_of_error)
    else:
        confidence_interval = (np.nan, np.nan)

    # Create quantile table
    quantiles = [i for i in range(100, 0, -10)]
    quantile_values = np.percentile(gains, quantiles) if gains else []
    quantile_table = pd.DataFrame({
        'Quantile': quantiles,
        'Gain': quantile_values
    })

    return gains, tickers_used, pd.DataFrame(transaction_table), average_gain, confidence_interval, quantile_table


def backtest_strategies_rsi(stock_df, start_test_date, end_test_date,
                            initial_capital, max_holding_days,
                            lags_buy_days, gain_threshold, use_end_date, trade_size,
                            rsi_entry, rsi_exit):
    """
    Backtest strategies using RSI-based trading rules.

    Args:
        stock_df (DataFrame): Stock data.
        start_test_date (str): Start date for backtesting.
        end_test_date (str): End date for backtesting.
        initial_capital (float): Starting capital for the backtest.
        max_holding_days (int): Maximum holding period.
        lags_buy_days (int): Days to wait after RSI entry threshold before buying.
        gain_threshold (float): Minimum percentage gain required for selling.
        use_end_date (bool): Whether to use the end date of data for selling.
        trade_size (float): Trade size (not used in RSI logic but kept for consistency).
        rsi_entry (int): RSI threshold for buying.
        rsi_exit (int): RSI threshold for selling.

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

    # Precompute RSI
    stock_df = precompute_rsi(stock_df)

    transaction_cost = 0.001  # Transaction cost of 0.1%

    # Run individual RSI test
    gains, tickers_used, transaction_table, average_gain, confidence_interval, quantile_table = individual_rsi_test(
        stock_df, start_test_date, end_test_date,
        initial_capital, lags_buy_days, gain_threshold, max_holding_days,
        transaction_cost, use_end_date, rsi_entry, rsi_exit
    )

    tot_gains = sum(gains)
    median_gain = np.median(gains) if gains else 0
    selling_rule = f'lags_buy_days={lags_buy_days}, gain_threshold={gain_threshold}, max_holding_days={max_holding_days}'

    # Summary of results
    summary_results = pd.DataFrame({
        'Strategy': [
            'Individual RSI'
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


def backtest_multiple_stocks_rsi(transaction_table, initial_capital=10000):
    """
    Perform RSI backtest across multiple stocks using extracted transactions.

    Selling Rule:
        - Ensure buy-sell order is valid (cannot buy before selling another stock).
        - Use the sequence of transactions from the individual RSI backtest.

    Args:
        transaction_table (DataFrame): Individual RSI transactions.
        initial_capital (float): Starting capital for the backtest.

    Returns:
        total_gain (float): Total gain from the multiple stock backtest.
        avg_gain (float): Average gain per transaction.
        median_gain (float): Median gain across all trades.
        transaction_log (DataFrame): Log of valid multiple RSI transactions.
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


def execute_rsi_backtest(file_path, start_test_date, end_test_date, 
                         initial_capital, max_holding_days, 
                         gain_threshold, use_end_date, 
                         trade_size=1000, rsi_entry=30, rsi_exit=70, 
                         lags_buy_days=1):
    """
    Execute the RSI backtest with realistic trading logic.

    Args:
        file_path (str): Path to the input CSV file containing stock data.
        start_test_date (str): Start date of the backtest in 'YYYY-MM-DD' format.
        end_test_date (str): End date of the backtest in 'YYYY-MM-DD' format.
        initial_capital (float): Starting capital for the backtest.
        max_holding_days (int): Maximum holding period (in days).
        gain_threshold (float): Minimum percentage gain for selling (e.g., 0.05 for 5% gain).
        trade_size (float): Minimum capital allocation for each trade.
        rsi_entry (int): RSI entry threshold for buying (e.g., 30 for oversold).
        rsi_exit (int): RSI exit threshold for selling (e.g., 70 for overbought).
        lags_buy_days (int): Days to wait after RSI buy signal to confirm price increase.

    Returns:
        summary (DataFrame): Summary of backtest results for all strategies.
        transaction_table (DataFrame): Detailed log of all transactions.
        quantile_table (DataFrame): Quantile return table.
    """
    # Load stock data
    stock_df = pd.read_csv(file_path)

    # Run individual RSI backtest strategy
    summary, transaction_table, quantile_table = backtest_strategies_rsi(
        stock_df=stock_df,
        start_test_date=start_test_date,
        end_test_date=end_test_date,
        initial_capital=initial_capital,
        max_holding_days=max_holding_days,
        gain_threshold=gain_threshold,
        use_end_date=use_end_date,
        trade_size=trade_size,
        rsi_entry=rsi_entry,
        rsi_exit=rsi_exit,
        lags_buy_days=lags_buy_days
    )

    # Run multiple RSI backtest strategy
    total_gain, avg_gain, median_gain, multiple_transaction_log = backtest_multiple_stocks_rsi(
        transaction_table=transaction_table,
        initial_capital=initial_capital
    )

    # Create summary for multiple RSI backtest
    selling_rule = f'gain_threshold={gain_threshold}, max_holding_days={max_holding_days}, rsi_entry={rsi_entry}, rsi_exit={rsi_exit}, lags_buy_days={lags_buy_days}'
    multiple_rsi_summary = pd.DataFrame({
        'Strategy': ['Multiple Stocks RSI'],
        'Selling Rule': [selling_rule],
        'Median Gain': [median_gain],
        'Average Gain': [avg_gain],
        'Confidence': [None]
    })

    # Concatenate summaries
    summary = pd.concat([summary, multiple_rsi_summary], ignore_index=True)
    quantile_table['Selling Rule'] = selling_rule
    return summary, transaction_table, quantile_table

# Parameters
file_path = 'stock_fullsegment.csv'
start_test_date = "2019-10-01"
end_test_date = "2024-10-18"
max_holding_days = 100        # Reduced for shorter-term RSI strategy
gain_threshold = 0.0       # Lowered to capture smaller gains
initial_capital = 10000
trade_size = 1000
use_end_date = True
rsi_entry = 50               # Lowered to focus on oversold conditions
rsi_exit = 80                # Maintains a significant gap for rebounds
lags_buy_days = 0            # Shortened to act on signals faster

# Execute RSI backtest
summary, transaction_table, quantile_table = execute_rsi_backtest(
    file_path=file_path,
    start_test_date=start_test_date,
    end_test_date=end_test_date,
    initial_capital=initial_capital,
    max_holding_days=max_holding_days,
    gain_threshold=gain_threshold,
    use_end_date=use_end_date,
    trade_size=trade_size,
    rsi_entry=rsi_entry,
    rsi_exit=rsi_exit,
    lags_buy_days=lags_buy_days
)

# Print results
print(summary)
print(transaction_table)
print(quantile_table)

  

RULES = [
    (30, 70, 0, 50), (40, 75, 1, 80), (45, 80, 0, 100),
    (40, 80, 0, 50), (50, 80, 1, 100), (50, 80, 0, 80),
    (40, 70, 2, 100), (35, 80, 2, 100), (40, 90, 0, 60)
]

summary_df = pd.DataFrame([])
quantile_df = pd.DataFrame([])

for itv in RULES:
    rsi_entry = itv[0]
    rsi_exit = itv[1]
    lags_buy_days = itv[2]
    max_holding_days = itv[3]
    use_end_date = True
    file_path = r'C:\backupcgi\final_bak\stock_fullsegment.csv'
    start_test_date = "2019-10-01"
    end_test_date = "2024-10-18"
    initial_capital = 10000
    trade_size = 1000

    summary, transaction_table, quantile_table = execute_rsi_backtest(
        file_path=file_path,
        start_test_date=start_test_date,
        end_test_date=end_test_date,
        initial_capital=initial_capital,
        max_holding_days=max_holding_days,
        gain_threshold=gain_threshold,
        use_end_date=use_end_date,
        trade_size=trade_size,
        rsi_entry=rsi_entry,
        rsi_exit=rsi_exit,
        lags_buy_days=lags_buy_days
    )
    
    info = [rsi_entry, rsi_exit, lags_buy_days, max_holding_days]
    
    print (info)
    print ("**************************************")
    print (quantile_table)
    
    
    
