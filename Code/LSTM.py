# -----------------------------------------------
# Fatigue Classification using LSTM on EMG Features
# -----------------------------------------------

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load and Prepare Data
# -------------------------
# data = pd.read_csv('your_data_file.csv')
# Replace above with:
# data = pd.read_excel('path_to_your_excel_file.xlsx')

features = data.iloc[:, 2:182].values  # Columns 3 to 182
subject = data['Participant'].values
labels = data['RPE'].values
unique_subjects = np.unique(subject)

# -------------------------
# Window Creation Function
# -------------------------
def create_windows(features, labels, window_size=5):
    X_windows = []
    y_windows = []
    for i in range(len(features) - window_size + 1):
        window = features[i:i + window_size]
        window_labels = labels[i:i + window_size]
        if np.all(window_labels == window_labels[0]):
            X_windows.append(window)
            y_windows.append(window_labels[0])
    return np.array(X_windows), np.array(y_windows)

# -------------------------
# Normalize and Window by Subject
# -------------------------
window_size = 5
X_all, y_all, subj_all = [], [], []

for subj in unique_subjects:
    idx = (subject == subj)
    subj_features = features[idx]
    subj_labels = labels[idx]

    # Normalize
    scaler = StandardScaler()
    subj_features = scaler.fit_transform(subj_features)

    # Create consistent-label windows
    X, y = create_windows(subj_features, subj_labels, window_size)
    X_all.append(X)
    y_all.append(y)
    subj_all.extend([subj] * len(y))

X_windows = np.concatenate(X_all)
y_windows = np.concatenate(y_all)
subj_windows = np.array(subj_all)

# -------------------------
# Define LSTM Model
# -------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.4))
    model.add(LSTM(32, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # RPE classes 0–4
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# Leave-One-Subject-Out Evaluation
# -------------------------
accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

for test_subject in unique_subjects:
    test_idx = subj_windows == test_subject
    train_idx = ~test_idx

    X_train, y_train = X_windows[train_idx], y_windows[train_idx]
    X_test, y_test = X_windows[test_idx], y_windows[test_idx]

    # Impute missing values if necessary
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    X_test = imputer.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    # Build and train model
    model = build_lstm_model(input_shape=(window_size, X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.3, verbose=0)

    # Predict and evaluate
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    accuracy_list.append(acc)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)

    print(f"\nSubject {test_subject} Evaluation:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(classification_report(y_test, y_pred))

# -------------------------
# Summary Results
# -------------------------
print("\n\n===== Overall Performance =====")
print(f"Average Accuracy:  {np.mean(accuracy_list):.3f}")
print(f"Average Precision: {np.mean(precision_list):.3f}")
print(f"Average Recall:    {np.mean(recall_list):.3f}")
print(f"Average F1-score:  {np.mean(f1_list):.3f}")
