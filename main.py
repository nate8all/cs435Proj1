import pandas as pd
import os

#defining column names based on dataset
column_names = [
    "age", "workclass", "fnlwgt", "educationlvl", "education-num", "marital-status", "occupation", "relationship","race","sex","capital-gain",
    "capital-loss", "hours-per-week", "native-country", "income"
]

#define folder path where adult_data resides
folder_path = "adult"
file_path = os.path.join(folder_path, "adult.data")

#load dataset, update file path
adult_data = pd.read_csv(file_path, header = None, names = column_names, na_values = " ?")

#manually remove rows with missing vals
cleaned_data = adult_data.dropna()

#save cleaned dataset
cleaned_file_path = os.path.join(folder_path, "adult_cleaned.csv")
cleaned_data.to_csv(cleaned_file_path, index=False)
print("Cleaning complete. Missing values have been removed.")