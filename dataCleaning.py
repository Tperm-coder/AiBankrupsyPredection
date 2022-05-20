import pandas as pd
import numpy as np
from sklearn import preprocessing
# Reading the data
df = pd.read_csv("Bankruptcy.csv")

#Counting the number of null values
nullValues = df.isnull().sum()
nullValues.to_csv("cleanedData/nullValues.csv")

#Checking for duplicates
isDuplicated = df.duplicated()
isDuplicated.to_csv("cleanedData/duplicates.csv")
df.drop_duplicates()

# Counting the number of '?' characters in each column to drop undefined charachters 
undefinedCharactersCount = pd.read_csv("Bankruptcy.csv", index_col=0)
undefinedCharactersCount = undefinedCharactersCount.isin(['?']).sum(axis=0)
undefinedCharactersCount.to_csv("cleanedData/undefinedCharactersCount.csv")

# Dropping columns with high count of undefined characters 
df.drop(df.columns[[21, 37]], axis = 1, inplace = True)

# replace undefined characters with the column mean
for col in df :
    df[col] = df[col].replace('?',  np.mean(pd.to_numeric(df[col], errors='coerce')))

scaler = preprocessing.MinMaxScaler()
names = df.columns
d = scaler.fit_transform(df)
df = pd.DataFrame(d, columns=names)

df.to_csv("cleanedData/cleanedData.csv")
