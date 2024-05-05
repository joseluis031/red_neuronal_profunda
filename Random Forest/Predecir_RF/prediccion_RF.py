from sklearn.ensemble import RandomForestRegressor


class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=50)
    
    def entrenar(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predecir(self, X_prediccion):
        return self.model.predict(X_prediccion)
