import numpy as np
import pandas as pd
import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

@dataclass(init=True)
class ModelTrainerConfig:
    trained_model_filepath: str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_filepath = ModelTrainerConfig()
    def initiate_model_trainer(self,X_train_arr,y_train_arr,X_test_arr,y_test_arr):
        try:
            logging.info('Train and test data loaded')
            X_train = X_train_arr
            y_train = np.array(y_train_arr)
            X_test = X_test_arr
            y_test = np.array(y_test_arr)

            models={
                "Random Forest Classifier": RandomForestClassifier(),
                "Multinomial Naive Bayes": MultinomialNB()
            }

            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            logging.info("Evaluation of model completed")

            best_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]

            best_model = models[best_model_name]
            print(best_score)

        except Exception as e:
            raise CustomException(e,sys)