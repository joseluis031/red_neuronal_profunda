from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout



class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape):
        self.model = Sequential([
            Conv1D(64, 2, activation='relu', input_shape=input_shape),
            MaxPooling1D(1),
            Conv1D(64, 2, activation='relu'),
            MaxPooling1D(1),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(2)  # 2 neuronas de salida para predecir goles_equipo_local y goles_equipo_visitante
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def entrenar(self, X_train, y_train):
        self.history = self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    def evaluar(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predecir(self, X_prediccion):
        return self.model.predict(X_prediccion)
