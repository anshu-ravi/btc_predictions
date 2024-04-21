import xgboost as xgb
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt


class Evaluation:
    def evaluate_model(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        print(f'Val Metrics: RMSE: {rmse}, MAE: {mae}, MAPE: {mape}')
        return rmse, mae, mape

    def plot_predictions(self, y_train, y_test, y_pred, model_name):
        plt.figure(figsize=(8, 6))
        plt.plot(y_train.index, y_train, label='Train')
        plt.plot(y_test.index, y_test, label='Test')
        plt.plot(y_test.index, y_pred, label='Predicted')
        plt.legend()
        plt.title(f'Crypto Prices True vs Predicted for {model_name}')
        plt.show()

class XGBoostModel(Evaluation):
    def __init__(self, params=None):
        if not params:
            self.params = {
                "objective": "reg:squarederror",
                "eval_metric": ["mae", "mape","rmse"],
                "tree_method": "hist", 
                "max_depth": 4,
                "subsample": 0.5,
                "colsample_bytree": 0.8,
                "colsample_bynode": 0.4,
                "eta": 0.05,
            }
        else:
            self.params = params

    def convert_to_dmatrix(self, X, y):
        dmatrix = xgb.DMatrix(data=X, label=y)
        return dmatrix

    def train(self, X_train, y_train, X_test, y_test):

        dtrain = self.convert_to_dmatrix(X_train, y_train)
        dtest = self.convert_to_dmatrix(X_test, y_test)

        self.model = xgb.train(self.params, 
                               dtrain, 
                               evals=[(dtrain, "train"), (dtest, "test")], 
                               num_boost_round=1000, 
                               early_stopping_rounds=10)
    
        return self.model.best_iteration
    
    def predict(self, dmatrix, iteration):
        if isinstance(dmatrix, pd.DataFrame) or isinstance(dmatrix, np.ndarray):
            dmatrix = self.convert_to_dmatrix(dmatrix, None)
        return self.model.predict(dmatrix, iteration_range=(0, iteration))
    
    def save_model(self, filename):
        self.model.save_model(filename)
    
    def load_model(self, filename):
        self.model = xgb.Booster()
        self.model.load_model(filename)
        return self.model

class CatBoostModel(Evaluation):
    def __init__(self):
        self.params = {
            "objective": "RMSE",
            "eval_metric": ["mae", "mape","rmse"],
            "num_boost_round": 5000,
            "early_stopping_rounds": 100,
            "eta": 0.05,
            "reg_lambda": 2.5,
            "subsample": 0.5,
            "depth": 4,
            # "use-best-model": True,
        }
    
    def train(self, X_train, y_train, X_test, y_test):
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(X_train, y_train, eval_set=(X_test, y_test))
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, filename):
        self.model.save_model(filename)
    
    def load_model(self, filename):
        self.model = CatBoostRegressor()
        self.model.load_model(filename)
        return self.model
    
class LightGBMModel(Evaluation):
    def __init__(self):
        self.params = {
            "objective": "regression",
            "metric": ["mae", "mape","rmse"],
            "num_boost_round": 500000,
            "early_stopping_rounds": 1000,
            "learning_rate": 0.05,
            "lambda_l2": 2.5,
            "subsample": 0.5,
            "max_depth": 4,
        }
    
    def train(self, X_train, y_train, X_test, y_test):
        self.model = LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train, 
                       eval_set=(X_test, y_test),
                       )
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, filename):
        self.model.booster_.save_model(filename)
    
    def load_model(self, filename):
        self.model = LGBMRegressor()
        self.model.booster_ = self.model.booster_.create_model(filename)
        return self.model
        
        

