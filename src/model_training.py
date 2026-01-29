import os
import sys
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model_output_path = MODEL_OUTPUT_PATH
        self.processed_dir = PROCESSED_DIR
        self.processed_train_file = PROCESSED_TRAIN_FILE_PATH
        self.processed_test_file = PROCESSED_TEST_FILE_PATH

        os.makedirs(self.model_output_path, exist_ok=True)

    def initiate_model_training(self):
        try:
            with mlflow.start_run(run_name="XGBregressor") as run:
                
                
                logger.info("Starting our MLFLOW experimentation")
                
                logger.info("Loading processed data")
                train_df = pd.read_csv(os.path.join(self.processed_dir, "train.csv"))
                test_df = pd.read_csv(os.path.join(self.processed_dir, "test.csv"))
                
                logger.info("Logging the training and testing datset to MLFLOW")
                mlflow.log_artifact(self.processed_train_file, artifact_path="datasets")
                mlflow.log_artifact(self.processed_test_file , artifact_path="datasets")
                
                logger.info("Splitting data into training and testing sets")
                X_train = train_df.drop("Salary", axis=1)
                y_train = train_df["Salary"]
                X_test = test_df.drop("Salary", axis=1)
                y_test = test_df["Salary"]

                logger.info("Training Linear Regression model")
                model = XGBRegressor()
                model.fit(X_train, y_train)

                logger.info("Evaluating model")
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                logger.info("Logging the model into MLFLOW")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging Metrics to MLFLOW")
                
                mlflow.log_metric("r2_score",r2)
                mlflow.log_metric("mean_absolute_error",mae)
                mlflow.log_metric("mean_squared_error",mse)
                mlflow.sklearn.log_model(model, "model")
                
                logger.info(f"Model Training Performance: R2 Score: {r2}, MAE: {mae}, MSE: {mse}")
                print(f"Model Training Performance: R2 Score: {r2}, MAE: {mae}, MSE: {mse}")

                logger.info("Saving trained model")
                joblib.dump(model, os.path.join(self.model_output_path, "model.pkl"))
                
                run_id = run.info.run_id
                print("Experiment Run ID:", run_id)
                
                
                model_uri = f"runs:/{run_id}/model"
                model_name = "XGBREGRESSOR"

                client = MlflowClient()
                try:
                    client.create_registered_model(model_name)
                except:
                    pass  
                model_details = mlflow.register_model(model_uri, model_name)
                print("Registered Model:", model_details.name, "Version:", model_details.version)
                return r2

                
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise CustomException("Failed in model training", e)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()
