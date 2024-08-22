import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
file_path = askopenfilename(title="Select the CSV file containing trade data")

df = pd.read_csv(file_path)
num_simulations = 10000
simulated_returns = []
simulated_drawdowns = []

for _ in range(num_simulations):
    random_trade = df.sample(n=1, replace=True).iloc[0]
    simulated_returns.append(random_trade['Profit %'])
    simulated_drawdowns.append(random_trade['Drawdown %'])

simulation_results = pd.DataFrame({
    'Simulated Return': simulated_returns,
    'Simulated Drawdown': simulated_drawdowns
})

simulation_results.to_csv('monte_carlo_simulation_results.csv', index=False)

# Plotting
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
