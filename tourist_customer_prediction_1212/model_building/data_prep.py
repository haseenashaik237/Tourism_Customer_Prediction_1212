
# Import Necessary Libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi


DATASET_REPO_ID = "BujjiProjectPrep/Tourism-Customer-Prediction-1212"
REPO_TYPE = "dataset"

# Initializing HF API Client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Loading dataset from HF
DATASET_PATH = "hf://datasets/BujjiProjectPrep/Tourism-Customer-Prediction-1212/tourism.csv"
df = pd.read_csv(DATASET_PATH)


# Dropping unnecessary columns
df = df.drop(columns=[col for col in ['CustomerID', 'Unnamed: 0'] if col in df.columns])

# Defining Target Column
target_col = "ProdTaken"

# Dropping rows with NaN values in target and other features
df = df.dropna(subset=[target_col]).dropna().reset_index(drop=True)

# Splitting into train and test datasets
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="BujjiProjectPrep/Tourism-Customer-Prediction-1212",
        repo_type="dataset",
    )
