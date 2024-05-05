from sklearn.preprocessing import StandardScaler

class Scaler:
    def __init__(self):
        self.scaler = None
    
    def fit_transform(self, X_train):
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X_train)
    
    def transform(self, X_test):
        return self.scaler.transform(X_test)
