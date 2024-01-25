import sys

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import f1_score

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            test_score = f1_score(y_test,y_pred)
            report[list(model.keys())[i]] = test_score
            
        return report

    except Exception as e:
        raise CustomException(e,sys)