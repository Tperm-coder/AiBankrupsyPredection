import pandas as pd
import numpy as np

# replace empty cells and '?' with Null values
df = pd.read_csv("Bankruptcy.csv")
df = df.replace(r"", "NaN")
df = df.replace(r'?', "NaN")

# saving the new data
df.to_csv("cleanedData/cleaned_data.csv")

# reading the new data
df = pd.read_csv("cleanedData/cleaned_data.csv")

# perfoming linear interpolatation to fill the null values
df = df.interpolate()

# saving the new data
df.to_csv("cleanedData/cleaned_data_with_linear_interprolation.csv")

df = pd.read_csv("cleanedData/cleaned_data.csv")

# replace NaN values with the column mean
for col in df :
    mean = df[col].mean()
    df[col].fillna(mean, inplace = True)

df.to_csv("cleanedData/cleaned_data_with_mean_replacement.csv")

print("done")
