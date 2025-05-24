# -----------------------------------------------
# Fatigue Classification using Deep Neural Network
# -----------------------------------------------

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# -------------------------
# Import Libraries
# -------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Load Dataset
# -------------------------
file_path = '/content/drive/MyDrive/NCB-Armin/Fatigue analysis/MSc_Project/EMG_MainProject/Results/MLFeatures.xlsx'
data = pd.read_excel(file_path)

# Extract features and labels
features = data.iloc[:, 2:182].values  # Columns 3 to 182
subject = data['Participant'].values
labels = data['RPE'].values

# Normalize features within each participant
unique_subjects = np.unique(subject)
normalized_features = np.copy(features)

for subj in unique_subjects:
    indices = (subject == subj)
    scaler = StandardScaler()
    normalized_features[indices] = scaler.fit_transform(features[indices])

# Create normalized DataFrame
normalized_data_df = pd.DataFrame(normalized_features, columns=[f'Feature_{i}' for i in range(features.shape[1])])
normalized_data_df['Participant'] = subject
normalized_data_df['RPE'] = labels

# -------------------------
# Define the Neural Network Model
# -------------------------
def build_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dense(5, activation='softmax')  # RPE levels: 0 to 4
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------
# Leave-One-Subject-Out Evaluation
# -------------------------
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

input_dim = features.shape[1]
model = build_model(input_dim)

for test_subj in unique_subjects:
    train_data = normalized_data_df[normalized_data_df['Participant'] != test_subj]
    test_data = normalized_data_df[normalized_data_df['Participant'] == test_subj]

    X_train = train_data.drop(columns=['Participant', 'RPE']).values
    y_train = train_data['RPE'].values
    X_test = test_data.drop(columns=['Participant', 'RPE']).values
    y_test = test_data['RPE'].values

    # Handle missing values if any
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Train
    history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.3, verbose=0)

    # Predict
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f"\nSubject {test_subj} Evaluation:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(classification_report(y_test, y_pred))

# -------------------------
# Final Performance Summary
# -------------------------
print("\n\n===== Overall Performance =====")
print(f"Average Accuracy:  {np.mean(accuracy_list):.3f}")
print(f"Average Precision: {np.mean(precision_list):.3f}")
print(f"Average Recall:    {np.mean(recall_list):.3f}")
print(f"Average F1 Score:  {np.mean(f1_list):.3f}")
