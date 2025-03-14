import pandas as pd
import os
#import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
import numpy as np
from scipy.fftpack import dct 
#import scikit-learn
#print(sklearn.__version__)  #should print a version number


#defining column names based on dataset
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship","race","sex","capital-gain",
    "capital-loss", "hours-per-week", "native-country", "income"
]
wcolumn_names = [
    "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide",
    "density","pH","sulphates","alcohol","quality"
]

#define folder path where adult_data/test reside
folder_path = "adult"
data_file_path = os.path.join(folder_path, "adult.data")
test_file_path = os.path.join(folder_path, "adult.test")

#define wine folder and file path and load dataset 
#   needs to be worked on as the data needs to be turned numerical for PCA 
wfolder_path = "wine+quality"
redwine_file_path = 'wine+quality/winequality-red.csv'
redwine_data = pd.read_csv(redwine_file_path)
whitewine_file_path = 'wine+quality/winequality-white.csv'
whitewine_data = pd.read_csv(whitewine_file_path)

#load dataset, update file path
adult_data = pd.read_csv(data_file_path, header = None, names = column_names, na_values = " ?")
adult_test = pd.read_csv(test_file_path, header = None, names = column_names, na_values = " ?")

#manually remove rows with missing vals (question1)
cleaned_data = adult_data.dropna()
cleaned_test = adult_test.dropna()
print("Cleaning complete. Missing values from adult_data and adult_test have been removed.")
print(f"Shape of initial data set: {adult_data.shape}")
print(f"Shape of cleaned data set: {cleaned_data.shape}")
print(f"Shape of initial training set: {adult_test.shape}")
print(f"Shape of cleaned training set: {cleaned_test.shape}")

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
print("The following categorical columns were altered:", categorical_columns)

#PCA of cleaned_data
cleaned_data_centered= cleaned_data - cleaned_data.mean() #centers the data by subtracting the mean
cleaned_data_scaled = cleaned_data_centered / cleaned_data.std() #scales the data by dividing by the standarddeviation

pca = PCA(n_components=2)
cleaned_data_pca = pca.fit_transform(cleaned_data_scaled)

print("Explained variance ratio of each component of adult data:", pca.explained_variance_ratio_)

cleaned_data_pca_df = pd.DataFrame(cleaned_data_pca, columns=['PC1', 'PC2'])
cleaned_data_pca_file_path = os.path.join(folder_path, "cleaned_adult_data_PCAreduced.csv")
cleaned_data_pca_df.to_csv(cleaned_data_pca_file_path, index=False)

#PCA of cleaned_test (Numeric Columns only)
numeric_columns = cleaned_test.select_dtypes(include=[np.number]).columns
cleaned_test_centered = cleaned_test[numeric_columns] - cleaned_test[numeric_columns].mean() #centers the data by subtracting the mean
cleaned_test_scaled = cleaned_test_centered / cleaned_test[numeric_columns].std() #scales the data by dividing by the standard deviation
ctpca = PCA(n_components=2)
cleaned_test_pca = ctpca.fit_transform(cleaned_test_scaled)

print("Explained variance ratio of each component of adult test:", ctpca.explained_variance_ratio_)

cleaned_test_pca_df = pd.DataFrame(cleaned_test_pca, columns=['PC1', 'PC2'])
cleaned_test_pca_file_path = os.path.join(folder_path, "cleaned_adult_test_PCAreduced.csv")
cleaned_test_pca_df.to_csv(cleaned_test_pca_file_path, index=False)

#PCA red wine data
redwine_data_centered = redwine_data - redwine_data.mean()
redwine_data_scaled = redwine_data_centered / redwine_data.std()

rwpca = PCA(n_components=2)
redwine_pca = rwpca.fit_transform(redwine_data_scaled)

print("Explained variance ratio of each component of red wine", rwpca.explained_variance_ratio_)
redwine_pca_df = pd.DataFrame(redwine_pca, columns=['PC1', 'PC2'])
redwine_pca_filepath = os.path.join(wfolder_path, "redwine_reducedPCA.csv")
redwine_pca_df.to_csv(redwine_pca_filepath, index=False)

#subfunction for removing the categorical columns, question 5
#def remove_categorical_columns(df, categorical_cols):
#    return df.drop(columns=categorical_cols, errors='ignore')

# Remove categorical columns from the dataset
#cleaned_data = remove_categorical_columns(cleaned_data, categorical_columns)
#cleaned_test = remove_categorical_columns(cleaned_test, categorical_columns)