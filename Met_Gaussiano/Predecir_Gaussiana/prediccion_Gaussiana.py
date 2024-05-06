from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class GaussianProcessModel:
    def __init__(self, random_state=0, length_scale=2.0):
        self.random_state = random_state
        self.length_scale = length_scale
    
    def entrenar(self, X_train, y_train):
        kernel = 0.5 * RBF(length_scale=self.length_scale)
        self.modelo = GaussianProcessRegressor(kernel=kernel, random_state=self.random_state)
        self.modelo.fit(X_train, y_train)
    
    def predecir(self, X_prediccion):
        return self.modelo.predict(X_prediccion)
