
from xgboost import XGBRegressor

class PredictorXGBoost:
    def __init__(self):
        self.modelo = XGBRegressor()
    
    def entrenar_modelo(self, X_train, y_train):
        self.modelo.fit(X_train, y_train)
    
    def predecir(self, X_prediccion):
        return self.modelo.predict(X_prediccion)

