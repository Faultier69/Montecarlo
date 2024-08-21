import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Open file dialog to select the file
Tk().withdraw()  # Prevents Tkinter window from appearing
file_path = askopenfilename(title="Select the CSV file containing trade data")

# Load the data from the selected CSV file
df = pd.read_csv(file_path)

# Perform Monte Carlo simulation on individual trades
num_simulations = 10000
simulated_returns = []
simulated_drawdowns = []

for _ in range(num_simulations):
    # Randomly select a trade from the data
    random_trade = df.sample(n=1, replace=True).iloc[0]
    simulated_returns.append(random_trade['Profit %'])
    simulated_drawdowns.append(random_trade['Drawdown %'])

# Convert to DataFrame
simulation_results = pd.DataFrame({
    'Simulated Return': simulated_returns,
    'Simulated Drawdown': simulated_drawdowns
})

# Save results to CSV
simulation_results.to_csv('monte_carlo_simulation_results.csv', index=False)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(simulation_results['Simulated Return'], bins=50, alpha=0.7)
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.title('Distribution of Simulated Returns')

plt.subplot(1, 2, 2)
plt.hist(simulation_results['Simulated Drawdown'], bins=50, alpha=0.7)
plt.xlabel('Drawdown')
plt.ylabel('Frequency')
plt.title('Distribution of Simulated Drawdowns')

plt.tight_layout()
plt.show()
