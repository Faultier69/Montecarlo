import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Open file dialog to select the file
Tk().withdraw()  # Prevents Tkinter window from appearing
file_path = askopenfilename(title="Select the CSV file containing trade data")

# Load the data from the selected CSV file
df = pd.read_csv(file_path)

# Print the column names to inspect them
print(df.columns)
