import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "Crop_recommendation.csv")

def amplitude_encode(features):
    d = len(features)
    n_qubits = int(np.ceil(np.log2(d)))
    dim = 2 ** n_qubits
    vec = np.zeros(dim)
    vec[:d] = features
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_states = [amplitude_encode(x) for x in X_scaled]

with open(os.path.join(BASE_DIR, "quantum_knn_model1.pkl"), "wb") as f:
    pickle.dump({
        "train_states": train_states,
        "train_labels": y,
        "scaler": scaler,
        "class_names": list(le.classes_)
    }, f)

print("QKNN trained and saved ✅")