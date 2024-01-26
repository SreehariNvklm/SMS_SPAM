import numpy as np
import pandas as pd
import sys
import os

from src.exception import CustomException
from src.logger import logging

from sklearn.feature_extraction.text import TfidfVectorizer

class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_file_path:str = os.path.join('artifacts','preprocessor.pkl')

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
            sms_col = 'sms'
            X_train = train_data.drop(target_col,axis=1)
            y_train = train_data.drop(sms_col,axis=1)

            X_test = test_data.drop(target_col,axis=1)
            y_test = test_data.drop(sms_col,axis=1)


            logging.info("Train test split done")

            print(X_train)
            print(y_train)
            print(X_test)
            print(y_test)

            tfidf = TfidfVectorizer(stop_words="english")

            X_train_tfidf = tfidf.fit_transform(train_data.drop(target_col,axis=1))
            X_test_tfidf = tfidf.transform(X_test)

            logging.info('Applied TfIdf vectorizer on data')           
            
            logging.info("Data transformation done")
            return(
                X_train_tfidf,
                y_train.transpose(),
                X_test_tfidf,
                y_test.transpose()
            )
        except Exception as e:
            raise CustomException(e,sys)