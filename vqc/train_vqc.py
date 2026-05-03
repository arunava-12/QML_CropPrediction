import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from vqc_model import HybridVQC

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

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

model = HybridVQC(n_classes=len(le.classes_))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = loss_fn(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: {loss.item()}")

torch.save(model.state_dict(), os.path.join(BASE_DIR, "vqc_crop_model.pth"))

with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(BASE_DIR, "label_map.pkl"), "wb") as f:
    pickle.dump({label: i for i, label in enumerate(le.classes_)}, f)

print("VQC trained and saved ✅")