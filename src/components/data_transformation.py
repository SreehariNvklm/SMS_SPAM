import numpy as np
import pandas as pd
import sys
import os

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class DataTransformationConfig:
    preprocessor_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def initiate_data_transformation(self,train_data,test_data):
        try:
            logging.info('Data transformation initiated')
            train_data = pd.read_csv(train_data)
            test_data = pd.read_csv(test_data)

            logging.info('Read train and test set data')
            
            target_col = 'result'
            X_train = train_data.drop(target_col,axis=1)
            y_train = train_data[target_col]

            X_test = test_data.drop(target_col,axis=1)
            y_test = test_data[target_col]


            logging.info("Train test split done")

            tfidf = TfidfVectorizer(stop_words="english")

            tfidf.fit(X_train)
            X_train_tfidf = tfidf.transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)
            
            print(X_train_tfidf.shape)
            print(y_train.shape)

            logging.info('Applied TfIdf vectorizer on data')           
            
            logging.info("Data transformation done")
            return(
                X_train_tfidf,
                y_train,
                X_test_tfidf,
                y_test
            )
        except Exception as e:
            raise CustomException(e,sys)