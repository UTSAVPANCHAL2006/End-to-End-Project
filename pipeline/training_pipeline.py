from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTrainer
from utils.common_function import read_yaml
from config.paths_config import *

if __name__=="__main__":
    ### 1. Data Ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_FILE))
    data_ingestion.run()
    
    ### 2. Data Processing
    processor = DataProcessor(
        TRAIN_FILE_PATH,
        TEST_FILE_PATH,
        PROCESSED_DIR,
        CONFIG_FILE
    )
    processor.process_data()
    
    
    ### 3. Model Training
    
    trainer = ModelTrainer()
    trainer.initiate_model_training()