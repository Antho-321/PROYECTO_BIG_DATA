# -*- coding: utf-8 -*-
"""
Predicción de tipo de vino basado en características ingresadas por el usuario.
"""

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo previamente entrenado
model = load_model('wine_type_prediction_model.h5')

# Configurar las características seleccionadas
feature_names = ['alcohol', 'malic_acid', 'proline', 'flavanoids', 'od280/od315_of_diluted_wines']

# Inicializar el escalador (se debe usar el mismo escalador que se utilizó para entrenar el modelo)
scaler = MinMaxScaler(feature_range=(0, 1))
# Valores mínimos y máximos utilizados durante el entrenamiento (ajustar según los datos originales)
scaler.data_min_ = np.array([11.03, 0.74, 278.0, 0.34, 1.27])
scaler.data_max_ = np.array([14.83, 5.8, 1680.0, 5.08, 4.0])
scaler.scale_ = 1 / (scaler.data_max_ - scaler.data_min_)
scaler.min_ = -scaler.data_min_ * scaler.scale_

# Pedir al usuario que ingrese las características
print("Ingrese las siguientes características del vino:")
user_input = []
for feature in feature_names:
    value = float(input(f"{feature}: "))
    user_input.append(value)

# Normalizar las características ingresadas por el usuario
user_input = np.array(user_input).reshape(1, -1)
user_input_normalized = scaler.transform(user_input)

# Realizar la predicción
prediction = model.predict(user_input_normalized)
predicted_class = np.argmax(prediction, axis=-1)

# Mapear las clases a los nombres de los tipos de vino
wine_types = ['class_0', 'class_1', 'class_2']  # Reemplazar con nombres reales de las clases si están disponibles
predicted_wine_type = wine_types[predicted_class[0]]

# Mostrar el resultado
print("\n--- Resultado de la predicción ---")
print(f"El tipo de vino predicho es: {predicted_wine_type}")
