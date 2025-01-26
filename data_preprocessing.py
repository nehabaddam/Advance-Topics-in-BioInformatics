import pandas as pd
import numpy as np


import os
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt


from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Load each dataset
breast = pd.read_csv('breast.tsv', sep='\t')
cervical = pd.read_csv('cervical.tsv', sep='\t')
brain = pd.read_csv('brain.tsv', sep='\t')
lung = pd.read_csv('lung.tsv', sep='\t')


lung_columns = [col for col in lung.columns if lung[col].astype(str).str.contains('LUAD').any()] + [col for col in lung.columns if lung[col].astype(str).str.contains('Lung').any()] + ["Study ID", "Sample ID"]
lung_columns.remove("Cancer Type")
lung = lung.drop(lung_columns, axis=1)
breast_columns = [col for col in breast.columns if breast[col].astype(str).str.contains('BRCA').any()] + [col for col in breast.columns if breast[col].astype(str).str.contains('Breast').any()] + ["Study ID", "Sample ID"]
breast_columns.remove("Cancer Type")
breast = breast.drop(breast_columns, axis=1)
cervical_columns = [col for col in cervical.columns if cervical[col].astype(str).str.contains('CESC').any()] + [col for col in cervical.columns if cervical[col].astype(str).str.contains('Cervical').any()] + ["Study ID", "Sample ID"]
cervical_columns.remove("Cancer Type")
cervical = cervical.drop(cervical_columns, axis=1)
brain_columns = [col for col in brain.columns if brain[col].astype(str).str.contains('GBM').any()] + [col for col in brain.columns if brain[col].astype(str).str.contains('Glioblastoma').any()] + ["Study ID", "Sample ID"]
brain_columns.remove("Cancer Type")
brain = brain.drop(brain_columns, axis=1)

#  Impute missing values with mean for each dataset
breast_imputed = breast.fillna(breast.mean())
cervical_imputed = cervical.fillna(cervical.mean())
brain_imputed = brain.fillna(brain.mean())
lung_imputed = lung.fillna(lung.mean())

# Impute missing values with mode for object-type columns for each dataset
breast_imputed = breast_imputed.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x, axis=0)
cervical_imputed = cervical_imputed.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x, axis=0)
brain_imputed = brain_imputed.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x, axis=0)
lung_imputed = lung_imputed.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x, axis=0)


combined_clinical = pd.concat([breast_imputed, cervical_imputed, brain_imputed, lung_imputed], axis=0)
print(combined_clinical.isnull().sum().sum())


# Drop columns with too many missing values
filtered_combined_clinical = combined_clinical.dropna(thresh=0.6 * len(combined_clinical), axis=1)

# Separate categorical and numerical columns
categorical_columns = filtered_combined_clinical.select_dtypes(include=['object']).columns
numerical_columns = filtered_combined_clinical.select_dtypes(exclude=['object']).columns

# Label encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_columns:
    filtered_combined_clinical[col] = label_encoder.fit_transform(filtered_combined_clinical[col])

# Impute missing values using KNN imputer
imputer = KNNImputer()
imputed_data = imputer.fit_transform(filtered_combined_clinical)

# Convert imputed data back to DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=filtered_combined_clinical.columns)

imputed_df.to_csv('processes.csv', index=False)

