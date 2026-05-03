import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pennylane as qml

n_qubits = 7
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(n_qubits))

kernel_qnode = qml.QNode(kernel_circuit, dev)

def quantum_kernel(x1, x2):
    probs = kernel_qnode(x1, x2)
    return probs[0]  # overlap

def load_qsvm_model(filepath="qsvm/qsvm.pkl"):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["class_names"]

def predict_qsvm(input_data, model, scaler, class_names):
    X = scaler.transform([input_data])
    pred = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0]) * 100
    return class_names[pred], confidence