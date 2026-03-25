import pandas as pd
import torch
import torch.nn as nn
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from qreupload_model import ReuploadClassifier

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "Crop_recommendation.csv")

# ---------------- LOAD DATA ----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

X = df.drop("label", axis=1).values
y = df["label"].values

# ---------------- ENCODE LABELS ----------------
le = LabelEncoder()
y = le.fit_transform(y)
n_classes = len(le.classes_)

# ---------------- SCALE FEATURES ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TENSORS & DATALOADER ----------------
# Ensure dtype is float32 for consistency
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---------------- MODEL SETUP ----------------
model = ReuploadClassifier(n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

# ---------------- TRAINING LOOP ----------------
EPOCHS = 50 

print(f"Starting training for {n_classes} classes...")

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), os.path.join(BASE_DIR, "model.pth"))

# ---------------- SAVE PREPROCESSING ----------------
with open(os.path.join(BASE_DIR, "data.pkl"), "wb") as f:
    pickle.dump({
        "scaler": scaler,
        "class_names": list(le.classes_)
    }, f)

print("\nReupload model trained and saved successfully! ✅")