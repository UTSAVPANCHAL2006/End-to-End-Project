import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_function import read_yaml, load_data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process_data(self):
        try:
            logger.info("Starting Data PreProcessing")

            logger.info("Load Data")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            logger.info("Drop Duplicates")
            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()

            logger.info("Removing Missing Values")
            train_df = train_df.dropna()
            test_df = test_df.dropna()

            
            target_column_name = "Salary"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Column groups
            numeric_features = ["Age", "Years of Experience"]
            nominal_features = ["Gender", "Job Title"]
            ordinal_features = ["Education Level"]

            education_order = [["Bachelor's", "Master's", "PhD"]]

            # Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_features),
                    ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features),
                    ("ord", OrdinalEncoder(categories=education_order), ordinal_features),
                ]
            )

            logger.info("Fitting preprocessor on training data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)


            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(preprocessor, "artifacts/preprocessor.pkl")
            logger.info("Saved preprocessor to artifacts/preprocessor.pkl")            
            
            # Extract feature names
            feature_names = preprocessor.get_feature_names_out()
            
            
            train_df_processed = pd.DataFrame(input_feature_train_arr, columns=feature_names)
            train_df_processed["Salary"] = target_feature_train_df.reset_index(drop=True)
            
            test_df_processed = pd.DataFrame(input_feature_test_arr, columns=feature_names)
            test_df_processed["Salary"] = target_feature_test_df.reset_index(drop=True)
            
            train_df_processed.to_csv(os.path.join(self.processed_dir, "train.csv"), index=False)
            test_df_processed.to_csv(os.path.join(self.processed_dir, "test.csv"), index=False)
            
            logger.info(f"Saved processed data to {self.processed_dir}")

            return train_df_processed, test_df_processed

        except Exception as e:
            logger.error(f"Error During Data Preprocessing: {e}")
            raise CustomException("Failed To Process Data", e)

if __name__ == "__main__":
    processor = DataProcessor(
        TRAIN_FILE_PATH,
        TEST_FILE_PATH,
        PROCESSED_DIR,
        CONFIG_FILE
    )
    processor.process_data()
