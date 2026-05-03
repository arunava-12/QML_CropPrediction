import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import pickle
import os

# ================= CONFIG =================
n_qubits = 7
n_layers = 4
num_classes = 22

dev = qml.device("default.qubit", wires=n_qubits)

# ================= QUANTUM CIRCUIT =================
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# TorchLayer wrapper (REMOVES LOOP 🚀)
weight_shapes = {"weights": (n_layers, n_qubits, 3)}
q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


# ================= MODEL =================
class QNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.projector = nn.Linear(7, n_qubits)

        self.q_layer = q_layer

        # Improved classical head
        self.fc1 = nn.Linear(n_qubits, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.projector(x)

        # 🚀 No loop anymore
        x = self.q_layer(x)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))

        return self.fc3(x)


# ================= LOAD MODEL =================
def load_qnn_model(model_path="qnn/qnn1_model.pth",
                   data_path="qnn/encoded_data.pkl"):

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    model = QNNClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model, data['scaler'], data['label_encoder']


# ================= PREDICT =================
def predict_qnn(model, scaler, label_encoder, X_input, return_probs=False):

    if isinstance(X_input, np.ndarray):
        X_input = torch.tensor(X_input, dtype=torch.float32)

    X_scaled = torch.tensor(scaler.transform(X_input), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_scaled)

        probs = torch.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1).cpu().numpy()
        predicted_labels = label_encoder.inverse_transform(preds)

    if return_probs:
        return predicted_labels, probs.cpu().numpy()

    return predicted_labels