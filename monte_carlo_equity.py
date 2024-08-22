import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() 
file_path = askopenfilename(title="Select the CSV file containing trade data")

df = pd.read_csv(file_path)

num_simulations = 50
num_trades = len(df)  

equity_curves = []

for _ in range(num_simulations):
    simulated_trades = df.sample(n=num_trades, replace=True).reset_index(drop=True)
    cumulative_returns = simulated_trades['Profit %'].cumsum()
    equity_curves.append(cumulative_returns)

equity_curves_df = pd.DataFrame(equity_curves).transpose()

final_cumulative_returns = equity_curves_df.iloc[-1, :]

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(equity_curves_df, color='blue', alpha=0.3)
plt.xlabel('Trade Number')
plt.ylabel('Cumulative Return')
plt.title('Monte Carlo Simulation of Equity Curves')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.hist(final_cumulative_returns, bins=20, density=True, alpha=0.6, color='blue')
mu, std = norm.fit(final_cumulative_returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

plt.xlabel('Final Cumulative Return')
plt.ylabel('Density')
plt.title('Distribution of Final Cumulative Returns with Bell Curve')
plt.grid(True)

plt.tight_layout()
plt.show()

equity_curves_df.to_csv('monte_carlo_equity_curves.csv', index=False)
