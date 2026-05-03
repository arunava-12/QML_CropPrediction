import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "Crop_recommendation.csv")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel="rbf", probability=True)
model.fit(X_scaled, y)

with open(os.path.join(BASE_DIR, "qsvm.pkl"), "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "class_names": list(le.classes_)
    }, f)

print("QSVM trained and saved ✅")