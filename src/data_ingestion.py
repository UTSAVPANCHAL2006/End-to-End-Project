import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_function import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config['data_ingestion']
        self.local_file_path = self.config['local_file_path']
        self.train_ratio = self.config['train_ratio']
        
        os.makedirs(RAW_DIR,exist_ok=True)
        logger.info(f"Data Ingestion Started Using {self.local_file_path}")
    
    def ingest_data(self):
        try:
            logger.info("Reading Data From CSV")
            data = pd.read_csv("Data/Salary_Data.csv")

            
            logger.info("Splitting Data Into Train & Test")
            train_data, test_data = train_test_split(data,test_size=1 - self.train_ratio,random_state=42)
            
            train_data.to_csv(TRAIN_FILE_PATH,index=False)
            test_data.to_csv(TEST_FILE_PATH,index=False)
            
            logger.info(f"Training Data save to {TRAIN_FILE_PATH}")
            logger.info(f"Testing Data save to{TEST_FILE_PATH}")
            
            
        except Exception as e :
            logger.error("Error During Data Ingestion")
            raise CustomException("Failed To Ingest Data", e)
        
        
    def run(self):
        
        try:
            self.ingest_data()
            logger.info("Data Ingestion completed successfully")
            
        except Exception as e:
            raise e 
        
if __name__ =="__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_FILE))
    data_ingestion.run()