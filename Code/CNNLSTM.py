# -----------------------------------------------
# Fatigue Classification using CNN-LSTM on EMG Features
# -----------------------------------------------

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load and Prepare Data
# -------------------------
# Replace with your actual data loading line
# data = pd.read_excel('path_to_your_data.xlsx')

features = data.iloc[:, 2:182].values
subject = data['Participant'].values
labels = data['RPE'].values
unique_subjects = np.unique(subject)

# -------------------------
# Helper Function: Windowing
# -------------------------
def create_windows(features, labels, window_size):
    X, y = [], []
    for i in range(len(features) - window_size + 1):
        window = features[i:i + window_size]
        window_labels = labels[i:i + window_size]
        if np.all(window_labels == window_labels[0]):
            X.append(window)
            y.append(window_labels[0])
    return np.array(X), np.array(y)

# -------------------------
# Normalize + Create Windows for Each Subject
# -------------------------
window_size = 5
X_all, y_all, subj_all = [], [], []

for subj in unique_subjects:
    idx = subject == subj
    subj_features = StandardScaler().fit_transform(features[idx])
    subj_labels = labels[idx]
    X, y = create_windows(subj_features, subj_labels, window_size)
    X_all.append(X)
    y_all.append(y)
    subj_all.extend([subj] * len(y))

X_windows = np.concatenate(X_all)
y_windows = np.concatenate(y_all)
subj_windows = np.array(subj_all)

# -------------------------
# Define CNN-LSTM Model
# -------------------------
def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.4))
    model.add(LSTM(32, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------
# Leave-One-Subject-Out Training & Evaluation
# -------------------------
acc_list, prec_list, rec_list, f1_list = [], [], [], []

for test_subj in unique_subjects:
    test_idx = subj_windows == test_subj
    train_idx = ~test_idx

    X_train, y_train = X_windows[train_idx], y_windows[train_idx]
    X_test, y_test = X_windows[test_idx], y_windows[test_idx]

    # Impute missing values (if any)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    X_test = imputer.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    # Build and train model
    model = build_model(input_shape=(window_size, X_train.shape[2]))
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.3, verbose=0)

    # Predict and evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc_list.append(accuracy_score(y_test, y_pred))
    prec_list.append(precision_score(y_test, y_pred, average='weighted'))
    rec_list.append(recall_score(y_test, y_pred, average='weighted'))
    f1_list.append(f1_score(y_test, y_pred, average='weighted'))

    print(f"\nSubject {test_subj} Evaluation:")
    print(f"Accuracy:  {acc_list[-1]:.3f}")
    print(f"Precision: {prec_list[-1]:.3f}")
    print(f"Recall:    {rec_list[-1]:.3f}")
    print(f"F1 Score:  {f1_list[-1]:.3f}")
    print(classification_report(y_test, y_pred))

# -------------------------
# Summary Results
# -------------------------
print("\n\n===== Overall Performance =====")
print(f"Average Accuracy:  {np.mean(acc_list):.3f}")
print(f"Average Precision: {np.mean(prec_list):.3f}")
print(f"Average Recall:    {np.mean(rec_list):.3f}")
print(f"Average F1-score:  {np.mean(f1_list):.3f}")
