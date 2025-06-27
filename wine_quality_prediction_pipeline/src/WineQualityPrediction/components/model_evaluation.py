import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib

from src.WineQualityPrediction.entity.config_entity import ModelEvaluationConfig
from src.WineQualityPrediction.constants import *
from src.WineQualityPrediction.utils.common import read_yaml, create_directories,save_json

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/sanjaynreddy96/mlops-projects.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="sanjaynreddy96"
os.environ["MLFLOW_TRACKING_PASSWORD"]="ae8befe1b64d8c21fecc0f1f5931562aa4a2abc5"

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Tracking in MLFlow
        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Logging parameters and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Log model as artifact
            local_model_path = "artifacts/model_evaluation/sklearn_model"
            os.makedirs(local_model_path, exist_ok=True)
            joblib.dump(model, os.path.join(local_model_path, "model.joblib"))
            mlflow.log_artifacts(local_model_path, artifact_path="model")
            