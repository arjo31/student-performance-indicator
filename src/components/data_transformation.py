import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_features = ["writing score", "reading score"]
            categorical_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical Columns scaling completed")
            logging.info("Categorical Columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numerical_features),
                    ("categorical_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
            pass
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of training and testing data completed")

            logging.info("Obtaining preprocessing object..")
            preprocessing_obj = self.get_data_transformer()
            target_column = "math score"

            input_feature_train = train_df.drop(columns=[target_column], axis = 1)
            target_feature_train = train_df[target_column]

            input_feature_test = test_df.drop(columns=[target_column], axis = 1)
            target_feature_test = test_df[target_column]

            logging.info("Applying Preprocessing on train and test dataframe")

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_array = preprocessing_obj.fit_transform(input_feature_test)

            train_array = np.c_[
                input_feature_train_array, np.array(target_feature_train)
            ]
            test_array = np.c_[
                input_feature_test_array, np.array(target_feature_test)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Saved preprocessing pickle file")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)


