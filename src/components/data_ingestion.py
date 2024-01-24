import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

@dataclass(init=True)
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion initialization')
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            logging.info('Reading the dataset')
            df = pd.read_csv('notebook\data\spam.csv',encoding="ISO-8859-1")

            logging.info('Removing the unnecessary columns from the dataset')
            df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
            df = df.rename(columns={'v1':'result','v2':'sms'})

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Splitting the data into train and test data')
            train_set, test_set = train_test_split(df,test_size=.1,random_state=101)

            train_set.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,header=True,index=False)
            logging.info('Ingestion of data completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()