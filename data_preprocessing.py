# =========================================================
# 4.0 DATA CLEANING
# =========================================================

import pandas as pd

# Load dataset (use relative path for GitHub)
df = pd.read_csv("data/Gold Price.csv")

# Quick inspection
print("First 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# =========================================================
# 4.3 DATA COMBINATION (Holiday Feature Engineering)
# =========================================================

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('future.no_silent_downcasting', True)

# Ensure datetime format and sorting
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Holiday dates list
holiday_dates = [
    '2014-01-31','2015-02-19','2016-02-08','2017-01-28',
    '2018-02-16','2019-02-05','2020-01-25','2021-02-12',
    '2022-02-01','2023-01-22','2024-02-10','2025-01-29','2026-02-17',
    '2014-12-25','2015-12-25','2016-12-25','2017-12-25',
    '2018-12-25','2019-12-25','2020-12-25','2021-12-25',
    '2022-12-25','2023-12-25','2024-12-25','2025-12-25','2026-12-25'
]

holiday_dates = pd.to_datetime(holiday_dates)

# Create holiday flag
df['Is_Holiday'] = df['Date'].apply(
    lambda x: any(abs((x - d).days) <= 1 for d in holiday_dates)
)

# Day type feature
df['Day_Type'] = df['Is_Holiday'].map({
    True: 'Holiday',
    False: 'Normal Day'
})

# Financial features
df['Abs_Chg'] = df['Price'].pct_change().abs() * 100
df['Daily_Volatility'] = df['High'] - df['Low']

# Summary statistics
summary = df.groupby('Day_Type').agg({
    'Abs_Chg': 'mean',
    'Daily_Volatility': 'mean',
    'Volume': 'mean'
})

print("\n=== Summary Comparison ===")
print(summary)


# =========================================================
# FIGURE 5: GOLD PRICE + HOLIDAYS
# =========================================================

sns.set_style("whitegrid")

plt.figure(figsize=(15, 6))
plt.plot(df['Date'], df['Price'],
         color='#E0E0E0', linewidth=1,
         label='Daily Gold Price')

holidays = df[df['Is_Holiday']]
plt.scatter(holidays['Date'], holidays['Price'],
            color='red', s=30, label='Holiday')

plt.title("Gold Price Time Series with Holiday Events")
plt.legend()
plt.show()


# =========================================================
# FIGURE 6: VOLATILITY COMPARISON
# =========================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

order = ['Holiday', 'Normal Day']

# Absolute price change
sns.barplot(
    data=df,
    x='Day_Type',
    y='Abs_Chg',
    hue='Day_Type',
    palette='Oranges',
    order=order,
    ax=ax1,
    errorbar=None,
    legend=False
)

ax1.set_title("Average Absolute Price Fluctuation (%)")
ax1.set_xlabel("Day Type")
ax1.set_ylabel("|Price Change| (%)")

# Daily volatility
sns.barplot(
    data=df,
    x='Day_Type',
    y='Daily_Volatility',
    hue='Day_Type',
    palette='Blues',
    order=order,
    ax=ax2,
    errorbar=None,
    legend=False
)

ax2.set_title("Average Daily Price Range (High - Low)")
ax2.set_xlabel("Day Type")
ax2.set_ylabel("Volatility")

plt.suptitle("Market Volatility Comparison")
plt.tight_layout()
plt.show()


# =========================================================
# FIGURE 7: TRADING VOLUME COMPARISON
# =========================================================

plt.figure(figsize=(8, 6))

sns.barplot(
    data=df,
    x='Day_Type',
    y='Volume',
    hue='Day_Type',
    palette='viridis',
    errorbar=None,
    legend=False
)

plt.title("Average Trading Volume Comparison")
plt.xlabel("Day Type")
plt.ylabel("Average Volume")
plt.tight_layout()
plt.show()


# =========================================================
# 4.3 DATA TRANSFORMATION
# =========================================================

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Reload dataset (clean pipeline style)
df = pd.read_csv("data/Gold Price.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Clean price column
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Price'] = df['Price'].ffill()

# Normalization
scaler = MinMaxScaler()
df['Price_Normalized'] = scaler.fit_transform(df[['Price']])

# Log transformation
df['Price_Log'] = np.log(df['Price'])

print("\nTransformed data preview:")
print(df.head())
