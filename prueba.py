import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Lee el CSV
df = pd.read_csv('CSVS/temp2023_24.csv')

# Codificación de variables categóricas
label_encoder = LabelEncoder()
df['equipo_local'] = label_encoder.fit_transform(df['equipo_local'])
df['equipo_visitante'] = label_encoder.transform(df['equipo_visitante'])
df['fase'] = label_encoder.fit_transform(df['fase'])

# Divide los datos en conjuntos de entrenamiento y prueba
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


import torch
import torch.nn as nn

class PredictorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PredictorModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 3
hidden_size = 5
output_size = 2  # Salida para goles_equipo_local y goles_equipo_visitante

model = PredictorModel(input_size, hidden_size, output_size)


criterion = nn.MSELoss()  # Puedes ajustar la función de pérdida según tu problema
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Puedes ajustar la tasa de aprendizaje

# Entrenamiento del modelo
num_epochs = 100  # Puedes ajustar el número de épocas
for epoch in range(num_epochs):
    inputs = torch.tensor(train_data[['fase', 'equipo_local', 'equipo_visitante']].values, dtype=torch.float32)
    labels = torch.tensor(train_data[['goles_equipo_local', 'goles_equipo_visitante']].values, dtype=torch.float32)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Predicciones en el conjunto de prueba
with torch.no_grad():
    test_inputs = torch.tensor(test_data[['fase', 'equipo_local', 'equipo_visitante']].values, dtype=torch.float32)
    predictions = model(test_inputs).numpy()

# Convierte las predicciones a un DataFrame de pandas y descodifica las variables categóricas si es necesario
predictions_df = pd.DataFrame(predictions, columns=['goles_equipo_local', 'goles_equipo_visitante'])
print(predictions_df)
