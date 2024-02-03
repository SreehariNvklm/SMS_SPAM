import dill
import sys
import os

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import precision_score

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]            
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            test_score = precision_score(y_test,y_pred,pos_label='ham')
            report[list(models.keys())[i]] = test_score
            
        return report

    except Exception as e:
        raise CustomException(e,sys)

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def load_obj(file_path):
    try:
        with open(file_path,"rb") as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e,sys)