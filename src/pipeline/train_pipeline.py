from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.predict_pipeline import PredictPipeline

class TrainPipeline:
    def __init__(self):
        data_ingest = DataIngestion()
        train_data,test_data = data_ingest.initiate_data_ingestion()

        transformer = DataTransformation()
        X_train_array,y_train_array,X_test_array,y_test_array = transformer.initiate_data_transformation(train_data,test_data)

        predict_pipeline = PredictPipeline(X_train_array,y_train_array,X_test_array,y_test_array)

if __name__ == "__main__":
    trained_pipeline = TrainPipeline()