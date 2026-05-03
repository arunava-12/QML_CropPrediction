import pandas as pd
import torch
import torch.nn as nn
import pickle
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from qnn_model import QNNClassifier

# ================= PATH =================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "Crop_recommendation.csv")

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

X = df.drop("label", axis=1).values
y = df["label"].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# ================= DATALOADER =================
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ================= MODEL =================
model = QNNClassifier()

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)  # ✅ lower LR
loss_fn = nn.CrossEntropyLoss()

# ================= TRAIN =================
for epoch in range(30):
    total_loss = 0

    for xb, yb in loader:
        optimizer.zero_grad()

        outputs = model(xb)
        loss = loss_fn(outputs, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: {total_loss / len(loader):.4f}")

# ================= SAVE =================
torch.save(model.state_dict(), os.path.join(BASE_DIR, "qnn1_model.pth"))

with open(os.path.join(BASE_DIR, "encoded_data.pkl"), "wb") as f:
    pickle.dump({
        "scaler": scaler,
        "label_encoder": le
    }, f)

print("QNN trained and saved ✅")