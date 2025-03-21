import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA #for testing
import numpy as np
from scipy.fftpack import dct #for testing
from sklearn.preprocessing import StandardScaler 

# my built-in sklearn PCA and DCT functions have bneen commented out for implementation, 
# feel free to uncomment them to test for similarity

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
redwine_data = pd.read_csv(redwine_file_path, delimiter= ";")
whitewine_file_path = 'wine+quality/winequality-white.csv'
whitewine_data = pd.read_csv(whitewine_file_path, delimiter= ";")

#load dataset, update file path
adult_data = pd.read_csv(data_file_path, header = None, names = column_names, na_values = " ?")
adult_test = pd.read_csv(test_file_path, header = None, names = column_names, na_values = " ?")

#question 1, manually remove rows with missing vals
cleaned_data = adult_data.dropna()
cleaned_data = cleaned_data.drop("income", axis=1) #dropping target
cleaned_test = adult_test.dropna()
cleaned_test = cleaned_test.drop("income", axis=1)
print("Cleaning complete. Missing values from adult_data and adult_test have been removed. Target 'income' has been removed.")
print(f"Shape of initial data set: {adult_data.shape}")
print(f"Shape of cleaned data set: {cleaned_data.shape}\n")
print(f"Shape of initial training set: {adult_test.shape}")
print(f"Shape of cleaned training set: {cleaned_test.shape}\n")

#combine datasets before defining category lists
all_data = pd.concat([cleaned_data, cleaned_test], axis=0)

#define categorical columns to be converted/ ordinal encoding (question 2)
categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
all_data[categorical_columns] = all_data[categorical_columns].apply(lambda x: x.str.strip())
cleaned_data.loc[:, categorical_columns] = cleaned_data[categorical_columns].apply(lambda x: x.str.strip())
cleaned_test.loc[:, categorical_columns] = cleaned_test[categorical_columns].apply(lambda x: x.str.strip())

#automatically extract all unique categories for each categorical column, dynamically allocate them
category_lists = {col: sorted(all_data[col].dropna().unique()) for col in categorical_columns}

#apply ordinal encoding using the dynamically generated categories
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
print(" ")

#question 3 - PCA
def applyPCA(data, folder_path="folder", file_name="pca_output.csv"):
    data = data.astype(float) #data_array = np.asarray(data, dtype=float)

    # Standardize the data
    mean = np.mean(data, axis=0)
    data_standardized = data - mean
    std = np.std(data_standardized, axis=0, ddof=1)
    std[std == 0] = 1
    data_standardized /= std

    U, S, Vt = np.linalg.svd(data_standardized, full_matrices=False)
    
    # Calculate explained variance ratio for num_components
    explained_variance_ratio = (S ** 2) / np.sum(S ** 2)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_variance >= 0.95) + 1

    components = Vt[:num_components]
    transformed_data = np.dot(data_standardized, components.T) #transform

    #Save to CSV
    pca_df = pd.DataFrame(transformed_data, columns=[f'PC{i+1}' for i in range(num_components)])
    pca_file_path = os.path.join(folder_path, file_name)
    pca_df.to_csv(pca_file_path, index=False)

    print(f"PCA complete. File saved as {file_name}")
    print(f"Reduced Dimensionality: {num_components} principal components, capturing {cumulative_variance[num_components - 1]*100:.4f}% variance\n")
    return transformed_data

def apply_sklearn_pca(data, folder_path="folder", file_name="pca_output.csv"):
    data = data.astype(float) #data_array = np.asarray(data, dtype=float)

    # Standardize the data
    mean = np.mean(data, axis=0)
    data_standardized = data - mean
    std = np.std(data_standardized, axis=0, ddof=1)
    std[std == 0] = 1
    data_standardized /= std

    pca = PCA()
    pca.fit(data_standardized)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_variance >= 0.95) + 1 # Keep 95% variance

    pca = PCA(n_components=num_components)
    transformed = pca.fit_transform(data_standardized)
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    
    pca_df = pd.DataFrame(transformed, columns=[f'PC{i+1}' for i in range(num_components)])
    pca_file_path = os.path.join(folder_path, file_name)
    pca_df.to_csv(pca_file_path, index=False)
    
    print(f"PCA complete using SKLEARN. File saved as {file_name}")
    print(f"Reduced dimensionality: {num_components} principal components, capturing {explained_variance:.2f}% variance\n")
    return transformed

#myDCT function does reflect a lot of the same values as the built-in implementation, although i wasn't able to properly transpose the values
#question3
def applyDCT(data, folder_path="folder", file_name="dct_output.csv"):
    data_array = np.asarray(data, dtype=float)

    #standardize the data
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    std[std == 0] = 1e-8  #Prevent division by zero
    data_standardized = (data_array - mean) / std

    num_samples, num_features = data.shape

    x = np.arange(num_features).reshape(-1, 1) #column
    y = np.arange(num_features).reshape(1, -1) #row/feature indices
    norm = np.where(y == 0, 1 / np.sqrt(num_features), np.sqrt(2/num_features)) #normalization factor
    cos_val = np.cos((2 * x + 1) * y * np.pi / (2 * num_features))

    #transformed_data = np.zeros_like(data_standardized)
    transformed_data = np.zeros((num_samples, num_features))

    #reduced_data = np.dot(data_standardized, features.T)

    for i in range(num_samples):
        for k in range(num_features):
            sum_val = 0.0
            for n in range(num_features):
                cos_val = np.cos(np.pi * k * (2 * n + 1) / (2 * num_features))
                sum_val += data_standardized[i, n] * cos_val
            
            if k == 0:
                transformed_data[i, k] = np.sqrt(1.0 / num_features) * sum_val
            else:
                transformed_data[i, k] = np.sqrt(2.0 / num_features) * sum_val
    
    #transformed_data = data @ (cos_val * norm)

    #find component num based on variance
    variance = np.var(transformed_data, axis=0)
    cumulative_variance = np.cumsum(variance) / np.sum(variance)
    num_components = np.argmax(cumulative_variance >= 0.95) + 1

    reduced_data = transformed_data[:, :num_components]
    
    dct_df = pd.DataFrame(reduced_data, columns=[f'DCT{i+1}' for i in range(num_components)])   
    dct_file_path = os.path.join(folder_path, file_name)
    dct_df.to_csv(dct_file_path, index=False)
    print(f"DCT complete. File saved as {file_name}")
    print(f"Reduced Dimensionality: {num_components} DCT components, capturing {cumulative_variance[num_components - 1]*100:.4f}% variance\n")
    return reduced_data
    
def apply_sklearn_dct(data, folder_path="folder", file_name="dct_output.csv"):
    data_array = np.asarray(data, dtype=float)
    
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_array)
    
    #apply DCT to the standardized data
    transformed_data = dct(data_standardized, type=2, axis=1, norm='ortho') #
    
    variance = np.var(transformed_data, axis=0)
    cumulative_variance = np.cumsum(variance) / np.sum(variance)
    num_components = np.argmax(cumulative_variance >= 0.95)
    
    transformed_data = transformed_data[:num_components, :]
    
    dct_df = pd.DataFrame(transformed_data.T, columns=[f'DCT{i+1}' for i in range(num_components)])
    dct_file_path = os.path.join(folder_path, file_name)
    dct_df.to_csv(dct_file_path, index=False)
    
    print(f"DCT complete using SKLEARN. File saved as {file_name}")
    print(f"Reduced Dimensionality: {num_components} DCT components, capturing {cumulative_variance[num_components - 1]*100:.4f}% variance\n")
    return transformed_data

applyPCA(cleaned_data, folder_path, "cleaned_data_PCA.csv")
applyPCA(cleaned_test, folder_path, "cleaned_test_PCA.csv")
applyPCA(redwine_data, wfolder_path, "redwine_PCA.csv")
applyPCA(whitewine_data, wfolder_path, "whitewine_PCA.csv")
#apply_sklearn_pca(cleaned_data, folder_path, "sklearn_data_PCA.csv")
#apply_sklearn_pca(cleaned_data, folder_path, "sklearn_test_PCA.csv")
#apply_sklearn_pca(redwine_data, wfolder_path, "sklearn_redwine_PCA.csv")
#apply_sklearn_pca(whitewine_data, wfolder_path, "sklearn_whitewine_PCA.csv")

applyDCT(cleaned_data, folder_path, "cleaned_data_DCT.csv")     #returns similar values
applyDCT(cleaned_test, folder_path, "cleaned_test_DCT.csv")
applyDCT(redwine_data, wfolder_path, "redwine_DCT.csv")
applyDCT(whitewine_data, wfolder_path, "whitewine_DCT.csv")
#apply_sklearn_dct(cleaned_data, folder_path, "sklearn_data_DCT.csv")
#apply_sklearn_dct(redwine_data, wfolder_path, "sklearn_redwine_DCT.csv")

#question 4 answered in doc

#question 5
#subfunction for removing the categorical columns
def remove_categorical_columns(df, categorical_cols):
    return df.drop(columns=categorical_cols, errors='ignore')

#removes categorical columns from the dataset
no_cat_data = remove_categorical_columns(cleaned_data, categorical_columns)
print(f"Categorical columns from cleaned_data dropped. Now called no_cat_data")
print(f"Shape of no_cat_data: {no_cat_data.shape}")

no_cat_test = remove_categorical_columns(cleaned_test, categorical_columns)
print(f"Categorical columns from cleaned_test dropped. Now called no_cat_test")
print(f"Shape of no_cat_test: {no_cat_test.shape}\n")

#apply PCA and DCT to the adult sets with no categorical columns
applyPCA(no_cat_data, folder_path, "no_cat_data_PCA.csv")
applyPCA(no_cat_test, folder_path, "no_cat_test_PCA.csv")
#apply_sklearn_pca(no_cat_data, folder_path, "sklearn_no_cat_data_PCA.csv")
#apply_sklearn_pca(no_cat_test, folder_path, "sklearn_no_test_data_PCA.csv")

#no cat DCT, returns 6 components
applyDCT(no_cat_data, folder_path, "no_cat_data_DCT.csv")     
applyDCT(no_cat_test, folder_path, "no_cat_test_DCT.csv")
#apply_sklearn_dct(no_cat_data, folder_path, "sklearn_no_cat_data_DCT.csv")
#apply_sklearn_dct(no_cat_test, folder_path, "sklearn_no_test_data_DCT.csv")


#writeup for question 5 included in doc

#question 6 - generate datasets to fail PCA/DCT
#since PCA takes a linear combo of principle components, if all components are the same, PCA fails
def failure_PCA(sample_num=100, dimensions=20, case="plain"):
    np.random.seed(42)                                          #ensures the same random dataset is generated upon running
    if case == "zero_variance_data":
        data = np.zeros((sample_num, dimensions))                #all values same, 1
    elif case == 'highly_correlated':
        base = np.random.rand(sample_num, 1)
        data = np.hstack([base * (i+1) for i in range(dimensions)])
    else:
        print("Invalid type of case")
    print(f"Generated dataset where PCA expected to fail. Shape: {data.shape}")
    return data

def failure_DCT(sample_num=100, dimensions=20, case="plain"):
    np.random.seed(42)
    if case == "constant_data":
        data = np.zeros((sample_num, dimensions)) #all values are the same
    else:
        print("Invalid type of case")
    print(f"Generated dataset where DCT expected to fail. Shape: {data.shape}")
    return data

fails_PCA_data = failure_PCA(case="zero_variance_data") #zero_variance_data random_data highly_correlated
print(fails_PCA_data)
applyPCA(fails_PCA_data, folder_path, "failed_PCA_data.csv")

fails_DCT_data = failure_DCT(case="constant_data") #constant_data
print(fails_DCT_data)
applyDCT(fails_DCT_data, folder_path, "failed_DCT_data.csv")