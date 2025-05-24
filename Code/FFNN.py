# ---------------------- Imports ----------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

# ---------------------- Load Data ----------------------
# Replace this with your actual Excel file path
file_path = 'drive/MyDrive/NCB-Armin/Fatigue analysis/MSc_Project/IMU_MainProject/Results/ML_extractedfeaturesdata.xlsx'

# Load the Excel file
data = pd.read_excel(file_path)

# Extract columns
features = data.iloc[:, 2:11].values  # Select feature columns
labels = data['Label'].values         # Fatigue label (0–4 or 0–1)
participant_numbers = data['Subject'].values  # Subject ID for LOSO

# ---------------------- Normalize Per Subject ----------------------
scaler = StandardScaler()
for subject_id in np.unique(participant_numbers):
    indices = np.where(participant_numbers == subject_id)
    features[indices] = scaler.fit_transform(features[indices])

# ---------------------- FFNN with LOSO Cross-Validation ----------------------
# Initialize performance metric lists
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Leave-One-Subject-Out cross-validation
logo = LeaveOneGroupOut()

# Loop over each fold
for fold, (train_index, test_index) in enumerate(logo.split(features, labels, participant_numbers)):
    print(f'\n--- Fold {fold + 1}/{len(np.unique(participant_numbers))} ---')
    
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Define a new FFNN model for each fold
    model = Sequential([
        Dense(100, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(len(np.unique(labels)), activation='softmax')  # Output for multi-class classification
    ])

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=9, batch_size=8, validation_data=(X_test, y_test), verbose=0)

    # Predict and evaluate
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Store results
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    # Print fold results
    print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}')

# ---------------------- Final Report ----------------------
print("\n=== Final Average Metrics Across All Folds ===")
print(f'Average Accuracy : {np.mean(accuracy_list):.3f}')
print(f'Average Precision: {np.mean(precision_list):.3f}')
print(f'Average Recall   : {np.mean(recall_list):.3f}')
print(f'Average F1 Score : {np.mean(f1_list):.3f}')
