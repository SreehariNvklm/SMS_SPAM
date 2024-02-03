import pandas as pd
import sys
import os
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.utils import load_obj

class PredictPipeline:
    def __init__(self,X_train,y_train,X_test,y_test):
        model_trainer = ModelTrainer()
        self.model = model_trainer.initiate_model_trainer(X_train,y_train,X_test,y_test)
        
class CustomData:
    def __init__(self,msg:str):
        self.msg = msg
    def get_data_frame(self):
        try:
            custom_data_dict = {"sms":[self.msg]}
            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            raise CustomException(e,sys)
    def predict(self,msg):
        try:     
            preprocessor = load_obj(os.path.join('artifacts','preprocessor.pkl'))
            model = load_obj(os.path.join('artifacts','model.pkl'))
            scaled_msg = preprocessor.transform(msg)
            prediction = model.predict(scaled_msg)
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)