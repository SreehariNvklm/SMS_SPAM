import sys

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            classifier = MultiOutputClassifier(model,n_jobs=-1)
            classifier.fit(X_train,y_train)
            y_pred = classifier.predict(X_test)
            test_score = f1_score(y_test,y_pred)
            report[list(models.keys())[i]] = test_score
            
        return report

    except Exception as e:
        raise CustomException(e,sys)