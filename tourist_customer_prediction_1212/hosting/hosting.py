from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourist_customer_prediction_1212/model_deployment",
    repo_id="BujjiProjectPrep/Tourism-Customer-Prediction-1212",
    repo_type="space",
    path_in_repo="",
)
