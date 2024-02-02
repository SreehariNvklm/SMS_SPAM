import sys

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