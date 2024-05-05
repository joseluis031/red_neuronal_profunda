from sklearn.linear_model import LinearRegression

class LinearRegressor:
    def __init__(self):
        self.modelo = LinearRegression()

    @staticmethod
    def entrenar_modelo(X_train, y_train):
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        return modelo

    @staticmethod
    def predecir(modelo, X_prediccion):
        return modelo.predict(X_prediccion)
