import pandas as pd
from sklearn.datasets import load_wine

# Cargar el dataset
wine_data = load_wine()

# Convertir a DataFrame
wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)

# Agregar la variable objetivo (target)
wine_df['target'] = wine_data.target

# Verificar si hay valores nulos
print(wine_df.isnull().sum())
