import pandas as pd
import numpy as np
from sklearn import preprocessing

#Applying feature scaling using normalization
def normalize_data(data_frame) :
    scaler = preprocessing.MinMaxScaler()
    names = data_frame.columns

    data_frame = pd.DataFrame(scaler.fit_transform(data_frame), columns=names)
    return data_frame

#Saving cleaned data in csv files
def get_file_name(method,is_normalized) :
    if (method == "mean") :
        if (is_normalized == True) :
            return "cleanedData/cleaned_data_with_mean_with_normalization_with_columns_dropped.csv"
        else :
            return "cleanedData/cleaned_data_with_mean_without_normalization_with_columns_dropped.csv"
       
# Replace empty cells and '?' with Null values
df = pd.read_csv("Bankruptcy.csv")
df = df.replace(r"", "NaN")
df = df.replace(r'?', "NaN")

#Replacing cells with null values with the mean of the attribute
def cleanData(method) :
    df = pd.read_csv("cleanedData/cleaned_data.csv")
    if (method == "mean") :
        for col in df :
            mean = df[col].mean()
            df[col].fillna(mean, inplace = True)

        file_name = get_file_name(method,False)
        df.to_csv(file_name)
    

        df = normalize_data(df)
        file_name = get_file_name(method,True)
        df.to_csv(file_name)


methods = ["mean"]
for method in methods :
    cleanData(method)
    
# saving the new data
df.to_csv("cleanedData/cleaned_data.csv")


    
print("done")
