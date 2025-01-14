# -*- coding: utf-8 -*-
"""
Created on [DATE]

@author: [YOUR NAME]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load the Wine dataset
wine_data = load_wine()
# Convert to DataFrame
wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

# View the first few rows of the dataset
print(wine_df.head(10))

# Descriptive statistics for numerical data
print(wine_df.describe())

# Class distribution
print(wine_df['target'].value_counts())

# Create a histogram for Alcohol
plt.figure(figsize=(8, 6))
sns.histplot(wine_df['alcohol'], bins=30, kde=True)
plt.xlabel('Alcohol')
plt.ylabel('Frequency')
plt.title('Alcohol Content Distribution')
plt.show()

# Bar plot of wine types
plt.figure(figsize=(8, 6))
sns.countplot(data=wine_df, x='target')
plt.xlabel('Wine Type')
plt.ylabel('Count')
plt.title('Distribution of Wine Types')
plt.xticks(ticks=[0, 1, 2], labels=wine_data.target_names)
plt.show()

# Pairplot for selected features
selected_features = ['alcohol', 'malic_acid', 'proline']
sns.pairplot(wine_df[selected_features + ['target']], hue='target', palette='viridis')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

# Box plot for Proline grouped by wine type
plt.figure(figsize=(10, 6))
sns.boxplot(data=wine_df, x='target', y='proline')
plt.xlabel('Wine Type')
plt.ylabel('Proline')
plt.title('Proline Content by Wine Type')
plt.xticks(ticks=[0, 1, 2], labels=wine_data.target_names)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = wine_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Wine Features')
plt.show()

# Filter and highlight significant correlations (above 0.6)
significant_mask = np.abs(correlation_matrix) < 0.6
correlation_matrix[significant_mask] = np.nan
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Significant Correlations (above 0.6)')
plt.show()

# Box plot of Alcohol
plt.figure(figsize=(10, 6))
sns.boxplot(x=wine_df['alcohol'])
plt.title('Box-Plot of Alcohol Content')
plt.xlabel('Alcohol')
plt.show()

# Detect and handle outliers in Alcohol
Q1 = wine_df['alcohol'].quantile(0.25)
Q3 = wine_df['alcohol'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_wine_df = wine_df[(wine_df['alcohol'] >= lower_bound) & (wine_df['alcohol'] <= upper_bound)]

# Box plot of Alcohol without outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=filtered_wine_df['alcohol'])
plt.title('Box-Plot of Alcohol Content (Without Outliers)')
plt.xlabel('Alcohol')
plt.show()
