# -*- coding: utf-8 -*-
"""
Entrenamiento de modelo para predicción de tipo de vino basado en 5 características seleccionadas.
"""

# Importar librerías necesarias
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, roc_curve, r2_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # pip install shap

# ---------- Cargar y procesar el dataset ----------

# Cargar el dataset de vino
wine_data = load_wine()

# Extraer características y etiquetas
features = ['alcohol', 'malic_acid', 'proline', 'flavanoids', 'od280/od315_of_diluted_wines']
selected_features_indices = [0, 1, 12, 6, 9]  # Índices de las características seleccionadas en el dataset
X = wine_data.data[:, selected_features_indices]
y = wine_data.target

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características numéricas
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- Construir y entrenar el modelo ----------

# Construir la red neuronal
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))  # Cambiar a softmax para clasificación multiclase

# Configurar el optimizador
opt = Adam(learning_rate=1e-3)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Configurar early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# ---------- Evaluar el modelo ----------

# Predicciones
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Reporte de clasificación
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine_data.target_names))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine_data.target_names, yticklabels=wine_data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Métricas adicionales
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")

# Curva ROC y AUC
plt.figure()
for i in range(3):  # Para cada clase
    fpr, tpr, _ = roc_curve((y_test == i).astype(int), (y_pred == i).astype(int))
    auc = roc_auc_score((y_test == i).astype(int), (y_pred == i).astype(int))
    plt.plot(fpr, tpr, label=f'Class {wine_data.target_names[i]} (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Learning curves
plt.figure()
plt.title('Loss / sparse_categorical_crossentropy')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.show()

plt.figure()
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.show()

# SHAP para explicaciones del modelo
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=features)

# Guardar el modelo
model.save('wine_type_prediction_model.h5')
print("Modelo guardado correctamente.")
