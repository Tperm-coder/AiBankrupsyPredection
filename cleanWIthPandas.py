import pandas as pd
import numpy as np
from sklearn import preprocessing

def normalize_data(data_frame) :
    scaler = preprocessing.MinMaxScaler()
    names = data_frame.columns

    data_frame = pd.DataFrame(scaler.fit_transform(data_frame), columns=names)
    return data_frame

def get_file_name(method,are_columns_droped,is_normalized) :
    if (method == "mean") :
        if (are_columns_droped == True) :
            if (is_normalized == True) :
                return "cleanedData/cleaned_data_with_mean_with_normalization_with_columns_dropped.csv"
            else :
                return "cleanedData/cleaned_data_with_mean_without_normalization_with_columns_dropped.csv"
        else :
            if (is_normalized == True) :
                return "cleanedData/cleaned_data_with_mean_with_normalization_without_columns_dropped.csv"
            else :
                return "cleanedData/cleaned_data_with_mean_without_normalization_without_columns_dropped.csv"

def cleanData(method,are_columns_droped) :
    df = pd.read_csv("cleanedData/cleaned_data.csv")
    if (method == "mean") :
        for col in df :
            mean = df[col].mean()
            df[col].fillna(mean, inplace = True)

        file_name = get_file_name(method,are_columns_droped,False)
        df.to_csv(file_name)
    

        df = normalize_data(df)
        file_name = get_file_name(method,are_columns_droped,True)
        df.to_csv(file_name)

# replace empty cells and '?' with Null values
df = pd.read_csv("Bankruptcy.csv")
df = df.replace(r"", "NaN")
df = df.replace(r'?', "NaN")

# saving the new data
df.to_csv("cleanedData/cleaned_data.csv")

methods = ["mean"]
for method in methods :
    cleanData(method,False)
    
print("done")
