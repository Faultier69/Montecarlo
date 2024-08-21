import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Open file dialog to select the file
Tk().withdraw()  # Prevents Tkinter window from appearing
file_path = askopenfilename(title="Select the CSV file containing trade data")

# Load the data from the selected CSV file
df = pd.read_csv(file_path)

# Perform Monte Carlo simulation on individual trades and create equity curves
num_simulations = 1000  # Number of equity curves to simulate
num_trades = len(df)   # Number of trades in your dataset

# Initialize a list to store all simulated equity curves
equity_curves = []

for _ in range(num_simulations):
    # Randomly shuffle the trades to simulate different sequences
    simulated_trades = df.sample(n=num_trades, replace=True).reset_index(drop=True)
    
    # Calculate the cumulative return (equity curve) for the simulated sequence
    cumulative_returns = simulated_trades['Profit %'].cumsum()
    
    # Store the equity curve
    equity_curves.append(cumulative_returns)

# Convert list of equity curves to DataFrame for plotting
equity_curves_df = pd.DataFrame(equity_curves).transpose()

# Calculate the final cumulative returns for each simulation (last value of each curve)
final_cumulative_returns = equity_curves_df.iloc[-1, :]

# Plot the simulated equity curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(equity_curves_df, color='blue', alpha=0.3)
plt.xlabel('Trade Number')
plt.ylabel('Cumulative Return')
plt.title('Monte Carlo Simulation of Equity Curves')
plt.grid(True)

# Plot the histogram of final cumulative returns with a bell curve
plt.subplot(1, 2, 2)
plt.hist(final_cumulative_returns, bins=20, density=True, alpha=0.6, color='blue')

# Fit a normal distribution to the data
mu, std = norm.fit(final_cumulative_returns)

# Plot the bell curve
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

# Optional: Save the equity curves to a CSV file
equity_curves_df.to_csv('monte_carlo_equity_curves.csv', index=False)
