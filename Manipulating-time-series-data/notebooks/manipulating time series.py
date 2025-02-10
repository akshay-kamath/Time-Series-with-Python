# IPython Notebook: Time Series Data Analysis with Pandas

# Introduction
# This notebook demonstrates time series analysis techniques using pandas

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chapter 1: Dates, Times, and Basic Time Series Functionality

# Date and Time Data Types
# Demonstrate the creation of `pd.Timestamp` objects [1].
from datetime import datetime
time_stamp = pd.Timestamp(datetime(2017, 1, 1))
print(time_stamp)

# Understands dates as strings [1]
time_stamp = pd.Timestamp('2017-01-01')
print(time_stamp)

# Show how to access attributes like year and day name [1].
print(time_stamp.year)
print(time_stamp.day_name())

# Illustrate the use of `pd.Period` objects and frequency attributes [2].
period = pd.Period('2017-01')
print(period)
print(period.freq)

# Demonstrate converting between `pd.Period` and `pd.Timestamp` [2].
period_to_timestamp = period.to_timestamp()
print(period_to_timestamp)

timestamp_to_period = period_to_timestamp.to_period('M')
print(timestamp_to_period)

# Sequences of Dates and Times
# Create `pd.DateTimeIndex` using `pd.date_range` with various frequencies [3].
index = pd.date_range(start='2017-1-1', periods=12, freq='M')
print(index)

# Convert `pd.DateTimeIndex` to `pd.PeriodIndex` [4].
period_index = index.to_period()
print(period_index)

# Create a time series DataFrame using `pd.DateTimeIndex` [4].
data = np.random.random((12, 2))
df = pd.DataFrame(data=data, index=index)
print(df.info())

# Working with Stock Prices
# Import a CSV file into a pandas DataFrame, and parse dates [5-7].
# The below code assumes you have a 'google.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    google = pd.read_csv('google.csv')
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    google = pd.DataFrame({'date': pd.to_datetime(['2015-01-02', '2015-01-05', '2015-01-06']),
                           'price': [524.81, 513.87, 501.96]})
    print("Created dummy google DataFrame.")

# Convert a column of strings to datetime64 using `pd.to_datetime` [6].
google['date'] = pd.to_datetime(google['date'])
print(google.info())

# Set the 'date' column as the index of the DataFrame [6].
google.set_index('date', inplace=True)
print(google.info())

# Demonstrate partial string indexing for selecting subperiods [8, 9].
if not google.empty:
    print(google['2015'].info())
    print(google['2015-3': '2016-2'].info())

# Demonstrate setting and changing `DateTimeIndex` frequency using `.asfreq()` [9, 10].
if not google.empty:
    daily_google = google.asfreq('D')
    print(daily_google.info())
    print(daily_google.head())

# Basic Time Series Calculations
# Demonstrate the use of `.shift()` to move data between past & future [11].
if not google.empty:
    google['shifted'] = google['price'].shift()
    print(google.head(3))

# Calculate one-period percent change [12].
if not google.empty:
    google['change'] = google['price'].div(google['shifted'])
    print(google[['price', 'shifted', 'change']].head(3))

# Demonstrate built-in time-series change using `.diff()` [13].
if not google.empty:
    google['diff'] = google['price'].diff()
    print(google[['price', 'diff']].head(3))

# Demonstrate built-in time-series percent change using `.pct_change()` [14].
if not google.empty:
    google['pct_change'] = google['price'].pct_change().mul(100)
    print(google[['price', 'pct_change']].head(3))

# Calculate multi-period returns [14, 15].
if not google.empty:
    google['return_3d'] = google['price'].pct_change(periods=3).mul(100)
    print(google[['price', 'return_3d']].head())

# Chapter 2: Time Series Transformations and Resampling

# Comparing Stock Performance
# Normalize price series to start at 100 [16, 17].
# The below code assumes you have a 'google.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    google = pd.read_csv('google.csv', parse_dates=['date'], index_col='date')
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    google = pd.DataFrame({'price': [313.06, 311.68, 303.83]},
                          index=pd.to_datetime(['2010-01-04', '2010-01-05', '2010-01-06']))
    print("Created dummy google DataFrame.")

if not google.empty:
    first_price = google['price'].iloc
    normalized = google['price'].div(first_price).mul(100)
    normalized.plot(title='Google Normalized Series')
    plt.show()

# Normalize multiple series using `.div()` [17-19].
# The below code assumes you have a 'stock_prices.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    prices = pd.read_csv('stock_prices.csv', parse_dates=['date'], index_col='date')
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    prices = pd.DataFrame({'AAPL': [30.57, 30.63, 30.41],
                           'GOOG': [313.06, 311.68, 303.83],
                           'YHOO': [17.10, 17.23, 17.17]},
                          index=pd.to_datetime(['2010-01-04', '2010-01-05', '2010-01-06']))
    print("Created dummy prices DataFrame.")

if not prices.empty:
    normalized = prices.div(prices.iloc)
    print(normalized.head(3))

# Compare stock performance against a benchmark [19-21].
# The below code assumes you have a 'benchmark.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    index = pd.read_csv('benchmark.csv', parse_dates=['date'], index_col='date')
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    index = pd.DataFrame({'SP500': [1132.99, 1136.52, 1137.14]},
                         index=pd.to_datetime(['2010-01-04', '2010-01-05', '2010-01-06']))
    print("Created dummy index DataFrame.")

if not prices.empty and not index.empty:
    prices = pd.concat([prices, index], axis=1).dropna()
    normalized = prices.div(prices.iloc).mul(100)
    normalized.plot()
    plt.show()

    # Plot the performance difference using `.sub()` [20, 21].
    tickers = ['AAPL', 'GOOG', 'YHOO']
    if all(ticker in normalized.columns for ticker in tickers):
        diff = normalized[tickers].sub(normalized['SP500'], axis=0)
        diff.plot()
        plt.show()

# Changing the Frequency: Resampling
# Create quarterly data using `pd.date_range` [22].
dates = pd.date_range(start='2016', periods=4, freq='Q')
data = range(1, 5)
quarterly = pd.Series(data=data, index=dates)
print(quarterly)

# Upsample from quarterly to monthly frequency using `.asfreq()` [23].
monthly = quarterly.asfreq('M')
print(monthly)

# Demonstrate filling missing values using `ffill`, `bfill`, and `fill_value` [23, 24].
monthly = monthly.to_frame('baseline')
monthly['ffill'] = quarterly.asfreq('M', method='ffill')
monthly['bfill'] = quarterly.asfreq('M', method='bfill')
monthly['value'] = quarterly.asfreq('M', fill_value=0)
print(monthly)

# Add missing months using `.reindex()` [25].
dates = pd.date_range(start='2016', periods=12, freq='M')
reindexed = quarterly.reindex(dates)
print(reindexed)

# Upsampling and Interpolation with .resample()
# The below code assumes you have a 'unrate.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    unrate = pd.read_csv('unrate.csv', parse_dates=['Date'], index_col='Date')
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    unrate = pd.DataFrame({'UNRATE': [4.0, 4.1, 4.0]},
                          index=pd.to_datetime(['2000-01-01', '2000-02-01', '2000-03-01']))
    print("Created dummy unrate DataFrame.")

# Demonstrate the use of `.resample()` for assigning frequency [26, 27].
if not unrate.empty:
    print(unrate.resample('MS').asfreq().info())

# Interpolate monthly real GDP growth [28, 29].
# The below code assumes you have a 'gdp.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    gdp = pd.read_csv('gdp.csv', parse_dates=['DATE'], index_col='DATE')
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    gdp = pd.DataFrame({'gdp': [1.2, 7.8, 5.2]},
                         index=pd.to_datetime(['2000-01-01', '2000-04-01', '2000-07-01']))
    print("Created dummy gdp DataFrame.")

if not gdp.empty:
    gdp_1 = gdp.resample('MS').ffill().add_suffix('_ffill')
    gdp_2 = gdp.resample('MS').interpolate().add_suffix('_inter')
    print(gdp_1.head())
    print(gdp_2.head())

    # Concatenate two DataFrames using `pd.concat()` [28, 29].
    df1 = pd.DataFrame([1, 2, 30], columns=['df1'])
    df2 = pd.DataFrame([3, 4, 31], columns=['df2'])
    concatenated = pd.concat([df1, df2], axis=1)
    print(concatenated)

    # Combine GDP growth & unemployment data [29].
    if not unrate.empty:
        combined = pd.concat([unrate, gdp_2], axis=1)
        combined.plot()
        plt.show()

# Downsampling and Aggregation
# The below code assumes you have a 'ozone.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    ozone = pd.read_csv('ozone.csv', parse_dates=['date'], index_col='date')
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    ozone = pd.DataFrame({'Ozone': [0.010443, 0.011817, 0.016810]},
                         index=pd.to_datetime(['2000-01-31', '2000-02-29', '2000-03-31']))
    print("Created dummy ozone DataFrame.")

# Create monthly ozone data using `.resample().mean()` and `.resample().median()` [32, 33].
if not ozone.empty:
    monthly_mean = ozone.resample('M').mean()
    monthly_median = ozone.resample('M').median()
    print(monthly_mean.head())
    print(monthly_median.head())

    # Use `.resample().agg()` for multiple aggregation functions [34].
    aggregated = ozone.resample('M').agg(['mean', 'std'])
    print(aggregated.head())

# Resample multiple time series [35].
# The below code assumes you have a 'ozone_pm25.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    data = pd.read_csv('ozone_pm25.csv', parse_dates=['date'], index_col='date')
    data = data.resample('D').asfreq()
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    data = pd.DataFrame({'Ozone': [0.005545, 0.016139, 0.017004],
                           'PM25': [20.800000, 6.500000, 8.493333]},
                          index=pd.to_datetime(['2000-01-31', '2000-02-29', '2000-03-31']))
    print("Created dummy data DataFrame.")

if not data.empty:
    data = data.resample('BM').mean()
    print(data.info())

# Chapter 3: Window Functions and Correlation

# Rolling Window Functions
# The below code assumes you have a 'google.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    data = pd.read_csv('google.csv', parse_dates=['date'], index_col='date')
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    data = pd.DataFrame({'price': [313.06, 311.68, 303.83, 313.53, 312.00]},
                          index=pd.to_datetime(['2010-01-04', '2010-01-05', '2010-01-06', '2010-01-07', '2010-01-08']))
    print("Created dummy data DataFrame.")

# Calculate a rolling average using `.rolling(window=30).mean()` [36, 37].
if not data.empty:
    rolling_mean = data['price'].rolling(window=30).mean()
    print(rolling_mean.head(30))

# Use offset-based window sizes [38].
    rolling_mean_offset = data['price'].rolling(window='30D').mean()
    print(rolling_mean_offset.head(30))

# Calculate multiple rolling metrics using `.agg()` [39].
    rolling_agg = data['price'].rolling('90D').agg(['mean', 'std'])
    rolling_agg.plot(subplots=True)
    plt.show()

# Calculate rolling quantiles [39].
    rolling = data['price'].rolling('360D')
    q10 = rolling.quantile(0.1).to_frame('q10')
    median = rolling.median().to_frame('median')
    q90 = rolling.quantile(0.9).to_frame('q90')
    quantiles = pd.concat([q10, median, q90], axis=1)
    quantiles.plot()
    plt.show()

# Expanding Window Functions
# Calculate expanding sum and cumulative sum [40].
df = pd.DataFrame({'data': range(5)})
df['expanding sum'] = df['data'].expanding().sum()
df['cumulative sum'] = df['data'].cumsum()
print(df)

# The below code assumes you have a 'sp500.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    data = pd.read_csv('sp500.csv', parse_dates=['date'], index_col='date')
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    data = pd.DataFrame({'SP500': [1515.73, 1530.77, 1536.03, 1533.70, 1540.98]},
                          index=pd.to_datetime(['2007-05-24', '2007-05-25', '2007-05-29', '2007-05-30', '2007-05-31']))
    print("Created dummy data DataFrame.")

# Calculate a running rate of return using `.pct_change()` and `.cumprod()` [41, 42].
if not data.empty:
    pr = data['SP500'].pct_change()
    pr_plus_one = pr.add(1)
    cumulative_return = pr_plus_one.cumprod().sub(1)
    cumulative_return.mul(100).plot()
    plt.show()

# Get the running min & max using `.expanding().min()` and `.expanding().max()` [42].
    data['running_min'] = data['SP500'].expanding().min()
    data['running_max'] = data['SP500'].expanding().max()
    data.plot()
    plt.show()

# Random Walks & Simulations
# Generate random numbers using `numpy.random.normal` [43].
from numpy.random import normal, seed
from scipy.stats import norm
seed(42)
random_returns = normal(loc=0, scale=0.01, size=1000)
sns.distplot(random_returns, fit=norm, kde=False)
plt.show()

# Create a random price path [43].
return_series = pd.Series(random_returns)
random_prices = return_series.add(1).cumprod().sub(1)
random_prices.mul(100).plot()
plt.show()

# Generate random S&P 500 returns [44, 45].
if not data.empty:
    from numpy.random import choice
    sample = data['SP500'].pct_change().dropna()
    n_obs = len(sample)
    random_walk = choice(sample, size=n_obs)
    random_walk = pd.Series(random_walk, index=sample.index)

    # Create random S&P 500 prices [45].
    start = data['SP500'].iloc
    sp500_random = pd.Series([start], index=[data.index]).append(random_walk.add(1).cumprod().mul(start))
    data['SP500_random'] = sp500_random
    data[['SP500', 'SP500_random']].plot()
    plt.show()

# Correlation & Relations Between Series
# The below code assumes you have a 'assets.csv' file in the same directory.
# If you don't have it, you can create a dummy DataFrame instead.
try:
    data = pd.read_csv('assets.csv', parse_dates=['date'], index_col='date').dropna()
except FileNotFoundError:
    # Create a dummy DataFrame if the file is not found
    data = pd.DataFrame({'sp500': [1530.77, 1536.03, 1533.70],
                           'nasdaq': [2599.38, 2607.73, 2602.26],
                           'bonds': [100.11, 100.22, 100.33],
                           'gold': [654.22, 652.11, 653.55],
                           'oil': [65.77, 66.11, 65.99]},
                          index=pd.to_datetime(['2007-05-25', '2007-05-29', '2007-05-30']))
    print("Created dummy data DataFrame.")

# Calculate daily returns using `.pct_change()` [46].
if not data.empty:
    daily_returns = data.pct_change().dropna()

    # Visualize pairwise linear relationships using `seaborn.jointplot` [46].
    sns.jointplot(x='sp500', y='nasdaq', data=daily_returns)
    plt.show()

    # Calculate all correlations using `.corr()` [46, 47].
    correlations = daily_returns.corr()
    print(correlations)

    # Visualize all correlations using `seaborn.heatmap` [47].
    sns.heatmap(correlations, annot=True)
    plt.show()

# Chapter 4: Building and Evaluating a Market Value-Weighted Index
# The below code assumes you have a 'listings.xlsx' and 'stocks.csv' file in the same directory.
# If you don't have them, you can create dummy DataFrames instead.

# Building a Cap-Weighted Index
# Load stock listing data [48, 49].
try:
    nyse = pd.read_excel('listings.xlsx', sheet_name='nyse', na_values='n/a')
    nyse.set_index('Stock Symbol', inplace=True)
    nyse.dropna(subset=['Sector'], inplace=True)
    nyse['Market Capitalization'] /= 1e6  # in Million USD
except FileNotFoundError:
    nyse = pd.DataFrame({
        'Company Name': ['Procter & Gamble Company (The)', 'Toyota Motor Corp Ltd Ord', 'ABB Ltd', 'Coca-Cola Company (The)',
                         'Wal-Mart Stores, Inc.', 'Exxon Mobil Corporation', 'J P Morgan Chase & Co', 'Johnson & Johnson',
                         'Alibaba Group Holding Limited', 'AT&T Inc.', 'Oracle Corporation', 'United Parcel Service, Inc.'],
        'Last Sale': [90.03, 104.18, 22.63, 42.79, 73.15, 81.69, 84.40, 124.99, 110.21, 40.28, 44.00, 103.74],
        'Market Capitalization': [230159.64, 155660.25, 48398.94, 183655.31, 221864.61, 338728.71, 300283.25, 338834.39,
                                   275525.00, 247339.52, 181046.10, 90180.89],
        'Sector': ['Consumer Non-Durables', 'Capital Goods', 'Capital Goods', 'Consumer Non-Durables', 'Consumer Services',
                   'Energy', 'Finance', 'Health Care', 'Miscellaneous', 'Public Utilities', 'Technology', 'Transportation']
    }, index=['PG', 'TM', 'ABB', 'KO', 'WMT', 'XOM', 'JPM', 'JNJ', 'BABA', 'T', 'ORCL', 'UPS'])
    print("Created dummy nyse DataFrame.")

# Select index components [50, 51].
if not nyse.empty:
    components = nyse.groupby(['Sector'])['Market Capitalization'].nlargest(1)
    components.sort_values(ascending=False, inplace=True)
    print(components)

    tickers = components.index.get_level_values('Stock Symbol').tolist()
    print(tickers)

    # Stock index components [51].
    columns = ['Company Name', 'Market Capitalization', 'Last Sale']
    component_info = nyse.loc[tickers, columns]
    pd.options.display.float_format = '{:,.2f}'.format
    print(component_info)

# Building the Value-Weighted Index
# Load historical stock prices [52-54].
try:
    data = pd.read_csv('stocks.csv', parse_dates=['Date'], index_col='Date').loc[:, tickers]
except (FileNotFoundError, KeyError):
    # Create a dummy DataFrame if the file is not found
    dates = pd.to_datetime(['2016-01-04', '2016-01-05', '2016-01-06'])
    data = pd.DataFrame({
        'PG': [90.03, 91.00, 92.00], 'TM': [104.18, 105.00, 106.00], 'ABB': [22.63, 23.00, 24.00],
        'KO': [42.79, 43.00, 44.00], 'WMT': [73.15, 74.00, 75.00], 'XOM': [81.69, 82.00, 83.00],
        'JPM': [84.40, 85.00, 86.00], 'JNJ': [124.99, 125.00, 126.00], 'BABA': [110.21, 111.00, 112.00],
        'T': [40.28, 41.00, 42.00], 'ORCL': [44.00, 45.00, 46.00], 'UPS': [103.74, 104.00, 105.00]
    }, index=dates)
    print("Created dummy data DataFrame.")

# Number of shares outstanding [54].
if not nyse.empty:
    shares = components['Market Capitalization'].div(components['Last Sale'])
    print(shares)

    # From stock prices to market value [54, 55].
    if not data.empty:
        no_shares = components['Market Capitalization'].div(components['Last Sale'])
        market_cap_series = data.mul(no_shares)

        # Aggregate market value per period [55, 56].
        agg_mcap = market_cap_series.sum(axis=1)

        # Value-based index [56].
        index = agg_mcap.div(agg_mcap.iloc).mul(100)
        index.plot(title='Market-Cap Weighted Index')
        plt.show()

# Evaluate Index Performance
# Value contribution by stock [57, 58].
if not nyse.empty and not data.empty:
    if 'market_cap_series' in locals():
        change = market_cap_series.iloc[-1] - market_cap_series.iloc
        print(change.sort_values())

        # Market-cap based weights [58].
        market_cap = components['Market Capitalization']
        weights = market_cap.div(market_cap.sum())
        print(weights.sort_values().mul(100))

        # Value-weighted component returns [59].
        index_return = (index.iloc[-1] / index.iloc - 1) * 100
        print(index_return)

        # Ensure weights and index_return are not empty
        if weights.any() and index_return != 0:
            weighted_returns = weights.mul(index_return)
            weighted_returns.sort_values().plot(kind='barh')
            plt.show()

        # Performance vs benchmark [59, 60].
        try:
            sp500 = pd.read_csv('sp500.csv', parse_dates=['Date'], index_col='Date')
            data = index.to_frame('Index')  # Convert pd.Series to pd.DataFrame
            data['SP500'] = sp500
            data.SP500 = data.SP500.div(data.SP500.iloc, axis=0).mul(100)

            # Performance vs benchmark: 30D rolling return [60].
            def multi_period_return(r):
                return (np.prod(r + 1) - 1) * 100

            data.pct_change().rolling('30D').apply(multi_period_return).plot()
            plt.show()

        except FileNotFoundError:
            print("sp500.csv file not found. Skipping benchmark comparison.")

# Additional Analysis
# Index components - price data [61].
# The below code was originally using DataReader from pandas_datareader which is now deprecated for google.
# For demonstration purposes, we will skip fetching data from the web and reuse previously loaded data.
# If you wish to fetch data from other sources, you need to install and use appropriate data readers.
if not data.empty:
    price_data = data  # Reuse the data DataFrame
else:
    price_data = pd.DataFrame({
        'ABB': [22.63, 23.00, 24.00], 'BABA': [110.21, 111.00, 112.00], 'JNJ': [124.99, 125.00, 126.00],
        'JPM': [84.40, 85.00, 86.00], 'KO': [42.79, 43.00, 44.00], 'ORCL': [44.00, 45.00, 46.00],
        'PG': [90.03, 91.00, 92.00], 'T': [40.28, 41.00, 42.00], 'TM': [104.18, 105.00, 106.00],
        'UPS': [103.74, 104.00, 105.00], 'WMT': [73.15, 74.00, 75.00], 'XOM': [81.69, 82.00, 83.00]
    }, index=pd.to_datetime(['2016-01-04', '2016-01-05', '2016-01-06']))
    print("Using dummy price data.")

# Index components: return correlations [62, 63].
if not price_data.empty:
    daily_returns = price_data.pct_change().dropna()
    correlations = daily_returns.corr()

    # Visualize the result as a heatmap [63].
    sns.heatmap(correlations, annot=True)
    plt.xticks(rotation=45)
    plt.title('Daily Return Correlations')
    plt.show()

    # Saving to a single Excel worksheet [63].
    try:
        correlations.to_excel(excel_writer='correlations.xls', sheet_name='correlations', startrow=1, startcol=1)
        print("Correlation data saved to correlations.xls")
    except Exception as e:
        print(f"Could not save to Excel (xls format). Error: {e}")

    # Saving to multiple Excel worksheets [63].
    try:
        price_data.index = price_data.index.date  # Keep only date component
        with pd.ExcelWriter('stock_data.xlsx') as writer:
            correlations.to_excel(excel_writer=writer, sheet_name='correlations')
            price_data.to_excel(excel_writer=writer, sheet_name='prices')
            daily_returns.to_excel(writer, sheet_name='returns')
        print("Price, returns and correlation data saved to stock_data.xlsx")
    except Exception as e:
        print(f"Could not save to Excel (xlsx format). Error: {e}")



