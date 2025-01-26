import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor
import joblib
from sklearn.impute import SimpleImputer
import tensorflow as tf

from tensorflow.keras import layers, models

# Load Data with Dask and optimize memory usage
data = dd.read_parquet('sequence_features_val.parquet')

# Drop 'ID' column if present and optimize data types
data = data.drop(columns=['ID'], errors='ignore').astype('float32')
data = data.compute()  # Convert to a Pandas DataFrame

# data = pd.concat([data.head(10000), data.tail(10000)]).reset_index(drop=True)

# Assign labels directly
num_positive_samples = 1249857
# num_positive_samples = 10000
labels = pd.Series([1] * num_positive_samples + [0] * (len(data) - num_positive_samples), name='label')

# data.to_parquet('sequence_features_val_20000.parquet', index=False, engine='pyarrow')

# Separate features and labels
features = data
assert len(features) == len(labels), "Mismatch between features and labels!"

features = features.fillna(0)

# Split into train and test sets with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    labels, 
    test_size=0.3, 
    random_state=42,
    shuffle=True,
    stratify=labels  
)

# Function for training and evaluating models
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Predictions': preds,
        'Trained_Model': model
    }

# Define models
models = [
    (RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42), 'Random Forest'),
    (XGBClassifier(n_estimators=100, tree_method='hist', random_state=42), 'XGBoost'),
    (MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42), 'Neural Network'),
]


results = []
with ThreadPoolExecutor() as executor:
    futures = []
    # Adding other models
    for model, name in models:  
        futures.append(executor.submit(train_and_evaluate, model, name))

    for future in futures:
        results.append(future.result())

# Compile results into a DataFrame
results_df = pd.DataFrame({
    'Model': [result['Model'] for result in results],
    'Accuracy': [result['Accuracy'] for result in results],
    'Precision': [result['Precision'] for result in results],
    'Recall': [result['Recall'] for result in results],
    'F1-Score': [result['F1-Score'] for result in results]
})

# Save results
results_df.to_csv('Final_results_ML_DL.csv', index=False)

# Save feature importances for Random Forest model
rf_importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': results[0]['Trained_Model'].feature_importances_  
})

rf_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
rf_importance_df.to_csv('feature_importances_ML_DL.csv', index=False)

# Save models
for result in results:
    joblib.dump(result['Trained_Model'], f"{result['Model'].replace(' ', '_').lower()}_model_all.joblib")

print("Models trained in parallel, results saved, and models serialized.")
