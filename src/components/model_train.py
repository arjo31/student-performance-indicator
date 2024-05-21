import numpy as np
import pandas as pd
import os 
import sys

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_arr[:, :-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "KNN" : KNeighborsRegressor(),
                "XGBoost" : XGBRegressor(),
                "Ada Boost" : AdaBoostRegressor(),
                "Cat Boost" : CatBoostRegressor(verbose=False),
                "Gradient Boost" : GradientBoostingRegressor(),
            }

            params = {
                "Linear Regression" : {},
                "Lasso":{
                    # 'alpha' : [1.0, 5.0, 10.0, 50.0,100.0],
                    'max_iter' : [100, 200, 500, 1000],
                    # 'selection' : ['cyclic', 'random']
                },
                "Ridge":{
                    # 'alpha' : [1.0, 5.0, 10.0, 50.0,100.0],
                    # 'max_iter' : [100, 200, 500, 1000],
                    # 'solver' : ['auto', 'cholesky', 'lsqr', 'sparse_cg','sag','saga','lbfgs']
                },
                "Random Forest" : {
                    'n_estimators': [8,16,32,64,128,256],
                    'criterion' : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    # 'max_features':['sqrt','log2',None]
                },
                "Decision Tree" : {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "KNN" : {
                    "n_neighbors" : [5,10,32,64,100,256],
                    "weights" : ['uniform', 'distance'],
                    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'p' : [1,2,4]
                },
                "XGBoost" : {
                    # 'booster' : ['gbtree', 'gblinear', 'dart'],
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    # 'gamma' : [.1,.5,.7,.9]
                },
                "Ada Boost" : {
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Cat Boost" : {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Gradient Boost" : {
                    # 'loss' : ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [8,16,32,64,128,256],
                    # 'criterion' : ['friedman_mse', 'squared_error'],
                    # 'max_features' : ['sqrt', 'log2', None],
                    # 'alpha' : [0.1,0.2,0.5,0.9],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                }
            }

            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, param = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if (best_model_score < 0.6):
                raise CustomException("No Best Model Found", sys)
            
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square =  r2_score(y_test, predicted)

            return r2_square
                
        except Exception as e:
            raise CustomException(e, sys)