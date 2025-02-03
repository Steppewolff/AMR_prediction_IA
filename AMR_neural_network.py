# Cargar módulos necesarios
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow.keras.models
from tensorflow.keras.layers import Dense, Dropout

# Paso 1: Cargar los datos
# Cargamos los datos de las cepas bacterianas
# Cambia 'training_data.csv' y 'test_data.csv' por los nombres reales de los archivos.
train_data = pd.read_csv('training_data.csv')  # 1200 cepas para entrenamiento
test_data = pd.read_csv('test_data.csv')      # 200 cepas para prueba

# Inspeccionar datos
print(train_data.head())

# Paso 2: Procesar los datos
# Variables independientes (mutaciones en genes)
X_train = train_data.iloc[:, :160].values  # Las primeras 160 columnas son mutaciones
X_test = test_data.iloc[:, :160].values

# Variables dependientes
y_train_resistance = train_data[['IMI', 'AZT', 'FEP']].values  # Resistencia/sensibilidad (MIC)
y_train_profile = train_data['danger_profile'].values          # Perfil de peligrosidad

# Codificar valores categóricos en las etiquetas
le_profile = LabelEncoder()
y_train_profile_encoded = le_profile.fit_transform(y_train_profile)

# Estandarizar los datos de entrada
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Paso 3: Definir el modelo de red neuronal
# Red neuronal para predecir resistencia/sensibilidad de antibióticos y perfil de peligrosidad
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # Salida para 3 antibióticos + perfil de peligrosidad
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Paso 4: Preparar las etiquetas para múltiples salidas
# Convertir las etiquetas en formato categórico (una-hot encoding)
y_train_combined = np.concatenate([
    pd.get_dummies(train_data['IMI']).values,
    pd.get_dummies(train_data['AZT']).values,
    pd.get_dummies(train_data['FEP']).values,
    pd.get_dummies(y_train_profile_encoded).values
], axis=1)

# Paso 5: Entrenar el modelo
history = model.fit(X_train_scaled, y_train_combined,
                    epochs=50, batch_size=32, validation_split=0.2)

# Paso 6: Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Postprocesar las predicciones para convertirlas en etiquetas legibles
predicted_profiles = le_profile.inverse_transform(np.argmax(y_pred[:, -3:], axis=1))

# Imprimir resultados
print("Predicciones del perfil de peligrosidad en el set de prueba:")
print(predicted_profiles)

# Generar reporte de clasificación para cada antibiótico
print("Reporte de clasificación para IMI, AZT, FEP:")
for i, antibiotic in enumerate(['IMI', 'AZT', 'FEP']):
    print(f"\nReporte para {antibiotic}:")
    print(classification_report(test_data[antibiotic], np.argmax(y_pred[:, i*2:(i+1)*2], axis=1)))