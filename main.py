import pandas as pd
import os
#import sklearn
from sklearn.preprocessing import OrdinalEncoder 
#import scikit-learn
#print(sklearn.__version__)  #should print a version number


#defining column names based on dataset
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship","race","sex","capital-gain",
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
print("Cleaning complete. Missing values from adult_data and adult_test have been removed.")

#combine datasets before defining category lists
all_data = pd.concat([cleaned_data, cleaned_test], axis=0)

#define categorical columns to be converted/ ordinal encoding (question 2)
categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "income"]
all_data[categorical_columns] = all_data[categorical_columns].apply(lambda x: x.str.strip())
cleaned_data.loc[:, categorical_columns] = cleaned_data[categorical_columns].apply(lambda x: x.str.strip())
cleaned_test.loc[:, categorical_columns] = cleaned_test[categorical_columns].apply(lambda x: x.str.strip())

# Automatically extract all unique categories for each categorical column, dynamically allocate them
category_lists = {col: sorted(all_data[col].dropna().unique()) for col in categorical_columns}

# Apply ordinal encoding using the dynamically generated categories
ordinal_encoder = OrdinalEncoder(categories=[category_lists[col] for col in categorical_columns],handle_unknown="use_encoded_value", unknown_value=-1)
ordinal_encoder.fit(all_data[categorical_columns])

cleaned_data.loc[:, categorical_columns] = ordinal_encoder.transform(cleaned_data[categorical_columns])
cleaned_test.loc[:, categorical_columns] = ordinal_encoder.transform(cleaned_test[categorical_columns])

#save cleaned, ordinal dataset
cleaned_data_file_path = os.path.join(folder_path, "adult_data_cleaned.csv")
cleaned_data.to_csv(cleaned_data_file_path, index=False)
cleaned_test_file_path = os.path.join(folder_path, "adult_test_cleaned.csv")
cleaned_test.to_csv(cleaned_test_file_path, index=False)

print("Processing complete. Ordinal numerical dataset saved to cleaned files.")
