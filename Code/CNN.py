import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models

# Load Data with Dask and optimize memory usage
data = dd.read_parquet('sequence_features_val.parquet')

# Drop 'ID' column if present and optimize data types
data = data.drop(columns=['ID'], errors='ignore').astype('float32')
data = data.compute()  # Convert to a Pandas DataFrame

# Assign labels directly
num_positive_samples = 1249857
labels = pd.Series([1] * num_positive_samples + [0] * (len(data) - num_positive_samples), name='label')

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
    stratify=labels  # Ensures both train and test have proportional classes
)

# Define CNN model using Keras
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Reshape((input_shape[1], 1), input_shape=(input_shape[1],)))  # Reshape for CNN
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train and evaluate the CNN model
def train_and_evaluate_cnn(X_train, X_test, y_train, y_test):
    cnn_model = create_cnn_model(X_train.shape)
    cnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=1)
    cnn_preds = cnn_model.predict(X_test)
    cnn_preds = (cnn_preds > 0.5).astype(int)  # Convert predictions to binary
    accuracy = accuracy_score(y_test, cnn_preds)
    precision = precision_score(y_test, cnn_preds)
    recall = recall_score(y_test, cnn_preds)
    f1 = f1_score(y_test, cnn_preds)
    
    return {
        'Model': 'CNN',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Predictions': cnn_preds,
        'Trained_Model': cnn_model
    }

# Run CNN training and evaluation directly
cnn_results = train_and_evaluate_cnn(X_train.values, X_test.values, y_train, y_test)

# Save results
results_df = pd.DataFrame({
    'Model': [cnn_results['Model']],
    'Accuracy': [cnn_results['Accuracy']],
    'Precision': [cnn_results['Precision']],
    'Recall': [cnn_results['Recall']],
    'F1-Score': [cnn_results['F1-Score']]
})
results_df.to_csv('Final_results_CNN_all.csv', index=False)

# Save CNN model
cnn_results['Trained_Model'].save('cnn_model_all.h5')

print("CNN model trained, results saved, and model serialized.")
