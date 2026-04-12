3.2.1
Figure 1: Summary Statistics of Gold Closing Price (2014–2026)
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Gold Price.csv")
df['Date'] = pd.to_datetime(df['Date'])

print("Figure 1:Summary Statistics of Gold Price (2014–2026)")
summary = df['Price'].describe()
summary_df = pd.DataFrame(summary).rename(columns={'Price':'Value'})
print(price_summary)

Figure 2: Daily Gold Price Trend from 2014 to 2026
plt.figure(figsize=(12,5))
plt.plot(df['Date'], df['Price'], color='gold')
plt.title("Figure 2: Daily Gold Price Trend from 2014 to 2026")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("figure2_gold_price_trend.png")
plt.show()

Figure 3: Distribution of Daily Gold Prices
plt.figure(figsize=(8,5))
plt.hist(df['Price'], bins=30, color='orange', edgecolor='black')
plt.title("Figure 3: Distribution of Daily Gold Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figure3_gold_price_distribution.png")
plt.show()

Figure 4: Minimum and Maximum Daily Gold Prices

min_price = df['Price'].min()
max_price = df['Price'].max()

plt.figure(figsize=(6,4))
plt.bar(['Minimum Price','Maximum Price'], [min_price, max_price], color=['green','red'])
plt.title("Figure 4: Minimum and Maximum Daily Gold Prices")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("figure4_gold_price_range.png")
plt.show()

