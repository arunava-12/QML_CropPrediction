import os
import pickle
import torch
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
import torch.nn as nn

# Paths relative to this file
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "vqc_crop_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.pkl")

# Quantum and model hyperparameters
n_qubits = 7
n_layers = 5
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# PennyLane device
dev = qml.device("default.qubit", wires=range(n_qubits))

def circuit(inputs, weights):
    AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# QNode wrapped for PyTorch
qnode = qml.QNode(circuit, dev, interface="torch")

class HybridVQC(nn.Module):
    def __init__(self, n_classes=None):
        super().__init__()
        # quantum feature extractor
        self.qlayer  = qml.qnn.TorchLayer(qnode, weight_shapes)
        # two-layer classical head
        self.hidden  = nn.Linear(n_qubits, 16)
        self.dropout = nn.Dropout(p=0.3)
        self.out     = nn.Linear(16, n_classes)

    def forward(self, x):
        q_out = self.qlayer(x)
        h     = torch.relu(self.hidden(q_out))
        h     = self.dropout(h)
        logits= self.out(h)
        return logits


def load_vqc_model():
    """
    Load the pretrained VQC model, scaler, and class names.
    Returns:
        model (HybridVQC), scaler (StandardScaler), class_names (list)
    """
    # Load preprocessing objects
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LABEL_MAP_PATH, "rb") as f:
        label_to_idx = pickle.load(f)
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    # Initialize and load model
    model = HybridVQC(n_classes=len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler, class_names


def predict_vqc(input_features, model, scaler, class_names, return_probs=False):
    """
    Returns:
        - prediction (default)
        - OR (prediction, probabilities) if return_probs=True
    """

    # ---- Preprocess ----
    X = scaler.transform([input_features])
    X_t = torch.tensor(X, dtype=torch.float32)

    # ---- Forward pass ----
    with torch.no_grad():
        logits = model(X_t)

        # ✅ Convert to probabilities
        probs = torch.softmax(logits, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        prediction = class_names[pred_idx]

    if return_probs:
        return prediction, probs.cpu().numpy()

    return prediction