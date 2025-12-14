
# Importing Necessary Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow


DATASET_REPO_ID = "BujjiProjectPrep/Tourism-Customer-Prediction-1212"
MODEL_REPO_ID = "BujjiProjectPrep/Tourism-Customer-Prediction-1212"
REPO_TYPE_DATASET = "dataset"
REPO_TYPE_MODEL = "model"
TARGET_COL = "ProdTaken"

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = "hf://datasets/BujjiProjectPrep/Tourism-Customer-Prediction-1212/Xtrain.csv"
Xtest_path  = "hf://datasets/BujjiProjectPrep/Tourism-Customer-Prediction-1212/Xtest.csv"
ytrain_path = "hf://datasets/BujjiProjectPrep/Tourism-Customer-Prediction-1212/ytrain.csv"
ytest_path  = "hf://datasets/BujjiProjectPrep/Tourism-Customer-Prediction-1212/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)[TARGET_COL]
ytest  = pd.read_csv(ytest_path)[TARGET_COL]

numeric_features = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=["object"]).columns.tolist()

# Class Weight to Handle Imbalance
class_counts = ytrain.value_counts()
if 0 in class_counts and 1 in class_counts:
    class_weight = class_counts[0] / class_counts[1]
else:
    class_weight = 1.0

# PreProcessing Pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    remainder="drop"
)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False
)

# Defining Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [3, 4, 5],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__subsample": [0.6, 0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.6, 0.8, 1.0],
}

# Create pipeline: preprocessing + model
model_pipeline = make_pipeline(preprocessor, xgb_model)


tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(https://electrophysiologic-unegregiously-christinia.ngrok-free.dev)

mlflow.set_experiment("tourist_customer_prediction_1212_experiments")


# 👉 Start main MLflow run
with mlflow.start_run():
    # (Optional) Log some high-level config
    mlflow.log_param("model", "xgboost")
    mlflow.log_param("scale_pos_weight", float(class_weight))
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("encoder", "OneHotEncoder")

    # Hyperparameter Tuning
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        scoring="recall",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(Xtrain, ytrain)

    # 🔁 Log each param combo as nested runs (like reference code)
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(
            run_name=f"grid_candidate_{i}",
            nested=True
        ):
            # Log the exact hyperparameters for this candidate
            mlflow.log_params(param_set)
            # Log mean and std of CV recall
            mlflow.log_metric("mean_test_recall", float(mean_score))
            mlflow.log_metric("std_test_recall", float(std_score))

    # 🎯 Log best parameters on the main run
    mlflow.log_params(grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluation on train & test
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)

    print("\n Training Classification Report:")
    print(classification_report(ytrain, y_pred_train))

    print("\n Test Classification Report:")
    print(classification_report(ytest, y_pred_test))

    train_report = classification_report(
        ytrain, y_pred_train, output_dict=True
    )
    test_report = classification_report(
        ytest, y_pred_test, output_dict=True
    )

    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)
    train_recall = recall_score(ytrain, y_pred_train)
    test_recall = recall_score(ytest, y_pred_test)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # 📊 Log metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_recall": float(train_recall),
        "test_recall": float(test_recall),
        "train_precision": float(train_report["1"]["precision"]),
        "train_f1": float(train_report["1"]["f1-score"]),
        "test_precision": float(test_report["1"]["precision"]),
        "test_f1": float(test_report["1"]["f1-score"]),
    })

    # Saving best model
    model_filename = "best_tourist_customer_xgb_model.joblib"
    joblib.dump(best_model, model_filename)

    # 📦 Log model file as MLflow artifact
    mlflow.log_artifact(model_filename, artifact_path="model")

    # 🚀 Uploading model to Hugging Face Model Hub
    MODEL_REPO_ID = "BujjiProjectPrep/Tourism-Customer-Prediction-1212"
    REPO_TYPE_MODEL = "model"

    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type=REPO_TYPE_MODEL)
        print(f"Model repo '{MODEL_REPO_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{MODEL_REPO_ID}' not found. Creating a new one...")
        create_repo(
            repo_id=MODEL_REPO_ID,
            repo_type=REPO_TYPE_MODEL,
            private=False
        )
        print(f"Model repo '{MODEL_REPO_ID}' created successfully.")

    api.upload_file(
        path_or_fileobj=model_filename,
        path_in_repo=model_filename,
        repo_id=MODEL_REPO_ID,
        repo_type=REPO_TYPE_MODEL,
    )
