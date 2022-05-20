import pandas as pd
import numpy as np
from sklearn import preprocessing

def normalize_data(data_frame) :
    scaler = preprocessing.MinMaxScaler()
    names = data_frame.columns

    data_frame = pd.DataFrame(scaler.fit_transform(data_frame), columns=names)
    return data_frame

# thank you copilot for the code
def get_file_name(method,are_columns_droped,is_normalized) :
    if (method == "meidan") :
        if (are_columns_droped == True) :
            if (is_normalized == True) :
                return "cleanedData/cleaned_data_with_median_with_normalization_with_columns_dropped.csv"
            else :
                return "cleanedData/cleaned_data_with_median_without_normalization_with_columns_dropped.csv"
        else :
            if (is_normalized == True) :
                return "cleanedData/cleaned_data_with_median_with_normalization_without_columns_dropped.csv"
            else :
                return "cleanedData/cleaned_data_with_median_without_normalization_without_columns_dropped.csv"
    elif (method == "mean") :
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
    elif (method == "mode") :
        if (are_columns_droped == True) :
            if (is_normalized == True) :
                return "cleanedData/cleaned_data_with_mode_with_normalization_with_columns_dropped.csv"
            else :
                return "cleanedData/cleaned_data_with_mode_without_normalization_with_columns_dropped.csv"
        else :
            if (is_normalized == True) :
                return "cleanedData/cleaned_data_with_mode_with_normalization_without_columns_dropped.csv"
            else :
                return "cleanedData/cleaned_data_with_mode_without_normalization_without_columns_dropped.csv"
    elif (method == "linear_interpolation") :
        if (are_columns_droped == True) :
            if (is_normalized == True) :
                return "cleanedData/cleaned_data_with_linear_interpolation_with_normalization_with_columns_dropped.csv"
            else :
                return "cleanedData/cleaned_data_with_linear_interpolation_without_normalization_with_columns_dropped.csv"
        else :
            if (is_normalized == True) :
                return "cleanedData/cleaned_data_with_linear_interpolation_with_normalization_without_columns_dropped.csv"
            else :
                return "cleanedData/cleaned_data_with_linear_interpolation_without_normalization_without_columns_dropped.csv"

   
    



def cleanData(method,are_columns_droped) :
    df = pd.read_csv("cleanedData/cleaned_data.csv")

    if (method == "meidan") :
        for col in df :
            median = df[col].median()
            df[col].fillna(median, inplace = True)

        file_name = get_file_name(method,are_columns_droped,False)
        df.to_csv(file_name)
    

        df = normalize_data(df)
        file_name = get_file_name(method,are_columns_droped,True)
        df.to_csv(file_name)

    elif (method == "mean") :
        for col in df :
            mean = df[col].mean()
            df[col].fillna(mean, inplace = True)

        file_name = get_file_name(method,are_columns_droped,False)
        df.to_csv(file_name)
    

        df = normalize_data(df)
        file_name = get_file_name(method,are_columns_droped,True)
        df.to_csv(file_name)


    elif (method == "mode") :
        for col in df :
            mode = df[col].mode()
            df[col].fillna(mode, inplace = True)

        file_name = get_file_name(method,are_columns_droped,False)
        df.to_csv(file_name)
    

        df = normalize_data(df)
        file_name = get_file_name(method,are_columns_droped,True)
        df.to_csv(file_name)

    
    elif (method == "linear_interpolation") :
        df = df.interpolate()
        file_name = get_file_name(method,are_columns_droped,False)
        df.to_csv(file_name)

        df = normalize_data(df)
        file_name = get_file_name(method,are_columns_droped,True)
        df.to_csv("cleanedData/cleaned_data_with_mean_with_linear_interpolation_with_normalization.csv")



# replace empty cells and '?' with Null values
df = pd.read_csv("Bankruptcy.csv")
df = df.replace(r"", "NaN")
df = df.replace(r'?', "NaN")

# saving the new data
df.to_csv("cleanedData/cleaned_data.csv")

methods = ["meidan","mean","linear_interpolation"]
for method in methods :
    cleanData(method,False)


# Dropping columns with high count of undefined characters 
df.drop(df.columns[[21, 37]], axis = 1, inplace = True)

# saving the new data
df.to_csv("cleanedData/cleaned_data.csv")
for method in methods :
    cleanData(method,True)



print("done")
