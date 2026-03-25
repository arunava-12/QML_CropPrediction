import torch
import torch.nn as nn
import pennylane as qml
import os
import numpy as np
import pickle

# ================= CONFIG =================
n_qubits = 7
n_layers = 4

dev = qml.device("default.qubit", wires=n_qubits)

# ================= QNODE =================
@qml.qnode(dev, interface="torch", diff_method="backprop")
def reupload_circuit(inputs, weights):
    for i in range(n_layers):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights[i:i+1], wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ================= MODEL =================
class ReuploadClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

        # 🔥 Stronger head
        self.fc1 = nn.Linear(n_qubits, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        outputs = []

        for xi in x:
            q_out = reupload_circuit(xi, self.weights)

            # ✅ KEEP GRADIENT
            if not isinstance(q_out, torch.Tensor):
                q_out = torch.stack(q_out)

            outputs.append(q_out.float())

        x = torch.stack(outputs)

        # 🔥 better learning
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)


# ================= LOAD =================
def load_qreupload_model():
    BASE_DIR = os.path.dirname(__file__)

    model_path = os.path.join(BASE_DIR, "model.pth")
    data_path = os.path.join(BASE_DIR, "data.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    model = ReuploadClassifier(len(data["class_names"]))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, data["scaler"], data["class_names"]


# ================= PREDICT =================
def predict_qreupload(input_data, model, scaler, class_names):
    X = torch.tensor(scaler.transform([input_data]), dtype=torch.float32)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs).item()
    confidence = torch.max(probs).item() * 100

    return class_names[pred], confidence