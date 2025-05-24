# ---------------------- Imports ----------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# ---------------------- Load Data ----------------------
file_path = 'drive/MyDrive/NCB-Armin/Fatigue analysis/MSc_Project/IMU_MainProject/Results/ML_extractedfeaturesdata.xlsx'
data = pd.read_excel(file_path)

features = data.iloc[:, 2:11].values  # Feature columns
labels = data['Label'].values         # Fatigue labels
subjects = data['Subject'].values     # Subject IDs

# ---------------------- Normalize Per Subject ----------------------
X = np.array(features).reshape(-1, 60)
y = np.array(labels)
z = np.array(subjects)

X_scaled = []
for subject_id in np.unique(subjects):
    subject_indices = np.where(subjects == subject_id)[0]
    scaler = StandardScaler()
    X_subject_scaled = scaler.fit_transform(X[subject_indices])
    X_scaled.append(X_subject_scaled)

X_scaled = np.vstack(X_scaled)

# ---------------------- Windowing ----------------------
sequence_length = 5
X_reshaped, y_reshaped, z_reshaped = [], [], []

for i in range(0, len(X_scaled) - sequence_length + 1, sequence_length - 4):
    X_seq = X_scaled[i:i + sequence_length]
    y_seq = y[i:i + sequence_length]
    z_seq = z[i:i + sequence_length]

    if all(label == y_seq[0] for label in y_seq):  # Keep only consistent labels
        X_reshaped.append(X_seq)
        y_reshaped.append(y_seq[0])
        z_reshaped.append(z_seq[0])

X_reshaped = np.array(X_reshaped)  # shape: (n_windows, 5, 60)
y_reshaped = np.array(y_reshaped)
z_reshaped = np.array(z_reshaped)

# ---------------------- LOSO LSTM Classification ----------------------
accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
logo = LeaveOneGroupOut()

fold = 0
for train_idx, test_idx in logo.split(X_reshaped, y_reshaped, z_reshaped):
    fold += 1
    print(f"\n--- Fold {fold}/{len(np.unique(z_reshaped))} ---")

    X_train, X_test = X_reshaped[train_idx], X_reshaped[test_idx]
    y_train, y_test = y_reshaped[train_idx], y_reshaped[test_idx]

    # Build model
    model = Sequential([
        LSTM(15, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        LSTM(8, return_sequences=True),
        LSTM(6),
        Dense(len(np.unique(y_reshaped)), activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30, batch_size=10, validation_data=(X_test, y_test), verbose=0)

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    accuracy_list.append(acc)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)

    print(f'Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1 Score: {f1:.3f}')

# ---------------------- Final Results ----------------------
print("\n=== Final Average Metrics Across All Folds ===")
print(f'Average Accuracy : {np.mean(accuracy_list):.3f}')
print(f'Average Precision: {np.mean(precision_list):.3f}')
print(f'Average Recall   : {np.mean(recall_list):.3f}')
print(f'Average F1 Score : {np.mean(f1_list):.3f}')
