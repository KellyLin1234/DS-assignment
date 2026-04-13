import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns
df = pd.read_csv('Gold Price.csv')
df['Date'] = pd.to_datetime (df['Date'])
df = df.sort_values('Date')
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Price', color='blue', label='Price') plt.title('Figure 2.11: Historical price trends of gold (2014-2026)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('figure_2_11_long_term_trend.png')
