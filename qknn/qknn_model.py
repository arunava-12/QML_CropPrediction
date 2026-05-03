import numpy as np
import pickle
import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# ---- Utility Functions ----

def amplitude_encode(features):
    d = len(features)
    n_qubits = int(np.ceil(np.log2(d)))
    dim = 2 ** n_qubits
    vec = np.zeros(dim)
    vec[:d] = features
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def build_state_prep(vec, n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.initialize(vec, range(n_qubits))
    return qc

def swap_test(vec1, vec2, shots=1024):
    n_qubits = int(np.log2(len(vec1)))
    qc = QuantumCircuit(1 + 2 * n_qubits, 1)
    qc.h(0)
    qc.compose(build_state_prep(vec1, n_qubits), qubits=range(1, 1 + n_qubits), inplace=True)
    qc.compose(build_state_prep(vec2, n_qubits), qubits=range(1 + n_qubits, 1 + 2 * n_qubits), inplace=True)
    for i in range(n_qubits):
        qc.cswap(0, i + 1, i + 1 + n_qubits)
    qc.h(0)
    qc.measure(0, 0)
    simulator = AerSimulator()
    tqc = transpile(qc, simulator)
    job = simulator.run(tqc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    prob_0 = counts.get('0', 0) / shots
    fidelity = 2 * prob_0 - 1
    return fidelity

# ---- Load & Predict Functions ----

def load_qknn_model(filepath='qknn/quantum_knn_model1.pkl'):
    """
    Load the QKNN model from a single pickle file.
    
    Returns:
        model: List of (quantum_state_vector, label_index)
        scaler: StandardScaler object
        class_names: List of class names
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file '{filepath}' not found.")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    train_states = data['train_states']
    train_labels = data['train_labels']
    scaler = data['scaler']
    class_names = data['class_names']

    model = list(zip(train_states, train_labels))
    return model, scaler, class_names


def predict_qknn(input_data, model, scaler, class_names, k=5, return_details=False):
    """
    Returns:
        - prediction
        - optionally confidence + neighbor info
    """

    # ---- Preprocess ----
    scaled_input = scaler.transform([input_data])[0]
    vec_test = amplitude_encode(scaled_input)

    # ---- Compute fidelities ----
    fidelities = [swap_test(vec_test, vec_train) for vec_train, _ in model]
    distances = [1 - f for f in fidelities]

    # ---- k nearest neighbors ----
    nn_idx = np.argsort(distances)[:int(k)]
    nearest_labels = [model[i][1] for i in nn_idx]
    nearest_fidelities = [fidelities[i] for i in nn_idx]

    # ---- Majority voting ----
    most_common_label = Counter(nearest_labels).most_common(1)[0][0]
    prediction = class_names[most_common_label]

    # ---- ✅ CONFIDENCE (REAL) ----
    confidence = max(0, min(100, np.mean(nearest_fidelities) * 100))

    if return_details:
        return prediction, confidence, nearest_labels, nearest_fidelities

    return prediction, confidence