from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

class Evaluator:
    @staticmethod
    def calcular_mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def calcular_r2_score(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def calcular_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)
    
