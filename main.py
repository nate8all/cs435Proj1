import pandas as pd
import os

#defining column names based on dataset
column_names = [
    "age", "workclass", "fnlwgt", "educationlvl", "education-num", "marital-status", "occupation", "relationship","race","sex","capital-gain",
    "capital-loss", "hours-per-week", "native-country", "income"
]

#define folder path where adult_data/test reside
folder_path = "adult"
data_file_path = os.path.join(folder_path, "adult.data")
test_file_path = os.path.join(folder_path, "adult.test")

#load dataset, update file path
adult_data = pd.read_csv(data_file_path, header = None, names = column_names, na_values = " ?")
adult_test = pd.read_csv(test_file_path, header = None, names = column_names, na_values = " ?")

#manually remove rows with missing vals (question1)
cleaned_data = adult_data.dropna()
cleaned_test = adult_test.dropna()

#save cleaned dataset
cleaned_data_file_path = os.path.join(folder_path, "adult_data_cleaned.csv")
cleaned_data.to_csv(cleaned_data_file_path, index=False)
cleaned_test_file_path = os.path.join(folder_path, "adult_test_cleaned.csv")
cleaned_test.to_csv(cleaned_test_file_path, index=False)

print("Cleaning complete. Missing values from adult_data and adult_test have been removed.")