import streamlit as st
import numpy as np
from PIL import Image
import os

from vqc.vqc_model import load_vqc_model, predict_vqc
from qnn.qnn_model import load_qnn_model, predict_qnn
from qknn.qknn_model import load_qknn_model, predict_qknn
from qsvm.qsvm_model import load_qsvm_model, predict_qsvm
from qreupload.qreupload_model import load_qreupload_model, predict_qreupload


# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    vqc_model, vqc_scaler, vqc_class_names = load_vqc_model()
    qnn_model, qnn_scaler, qnn_label_encoder = load_qnn_model()
    qknn_model, qknn_scaler, qknn_class_names = load_qknn_model()
    qsvm_model, qsvm_scaler, qsvm_classes = load_qsvm_model()
    qre_model, qre_scaler, qre_classes = load_qreupload_model()

    return (
        vqc_model, vqc_scaler, vqc_class_names,
        qnn_model, qnn_scaler, qnn_label_encoder,
        qknn_model, qknn_scaler, qknn_class_names,
        qsvm_model, qsvm_scaler, qsvm_classes,
        qre_model, qre_scaler, qre_classes
    )


(
    vqc_model, vqc_scaler, vqc_class_names,
    qnn_model, qnn_scaler, qnn_label_encoder,
    qknn_model, qknn_scaler, qknn_class_names,
    qsvm_model, qsvm_scaler, qsvm_classes,
    qre_model, qre_scaler, qre_classes
) = load_models()


# ================= TITLE =================
st.markdown("""
    <div style='text-align: center;'>
        <h1>🌱 <strong>CropQuest</strong> 🌱</h1>
        <h3>🌾 Quantum-Powered Crop Predictor 🌾</h3>
        <h4>Predict the Best Crop Based on Soil & Weather Conditions</h4>
    </div>
""", unsafe_allow_html=True)


# ================= CSS =================
st.markdown("""
<style>
.section-header {
    color: #3A8D4D;
    font-size: 24px;
    font-weight: bold;
    border-bottom: 2px solid #3A8D4D;
    padding-bottom: 5px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ================= MODEL SELECTION =================
st.markdown("<h3 style='text-align:center;'>Available Models</h3>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("VQC"):
        st.session_state.model_choice = "VQC"

with col2:
    if st.button("QNN"):
        st.session_state.model_choice = "QNN"

with col3:
    if st.button("QKNN"):
        st.session_state.model_choice = "QKNN"

with col4:
    if st.button("QSVM"):
        st.session_state.model_choice = "QSVM"

with col5:
    if st.button("ReUpload"):
        st.session_state.model_choice = "REUPLOAD"


if "model_choice" not in st.session_state:
    st.session_state.model_choice = "VQC"

st.success(f"Selected Model: {st.session_state.model_choice}")


# ================= INPUT =================
st.markdown('<div class="section-header">Soil Features</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
N = col1.number_input("Nitrogen", value=90.0)
P = col2.number_input("Phosphorus", value=42.0)
K = col3.number_input("Potassium", value=43.0)

st.markdown('<div class="section-header">Weather Features</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
temperature = col1.number_input("Temperature", value=20.0)
humidity = col2.number_input("Humidity", value=82.0)

col1, col2 = st.columns(2)
ph = col1.number_input("pH", value=6.5)
rainfall = col2.number_input("Rainfall", value=202.9)

features = [N, P, K, temperature, humidity, ph, rainfall]


# ================= IMAGE =================
def load_crop_image(crop):
    path = os.path.join("assets", f"{crop.lower()}.jpg")
    return Image.open(path) if os.path.exists(path) else None


# ================= MAIN =================
colA, colB = st.columns(2)
predict_clicked = colA.button("Predict Crop 🚀")
compare_clicked = colB.button("Compare All Models ⚔️")


# ================= PREDICT =================
if predict_clicked:

    choice = st.session_state.model_choice

    if choice == "VQC":
        pred, probs = predict_vqc(features, vqc_model, vqc_scaler, vqc_class_names, True)
        conf = np.max(probs) * 100

    elif choice == "QNN":
        arr = np.array(features).reshape(1, -1)
        pred, probs = predict_qnn(qnn_model, qnn_scaler, qnn_label_encoder, arr, True)
        pred = pred[0]
        conf = np.max(probs) * 100

    elif choice == "QKNN":
        pred, conf = predict_qknn(features, qknn_model, qknn_scaler, qknn_class_names)

    elif choice == "QSVM":
        pred, conf = predict_qsvm(features, qsvm_model, qsvm_scaler, qsvm_classes)

    elif choice == "REUPLOAD":
        pred, conf = predict_qreupload(features, qre_model, qre_scaler, qre_classes)

    st.markdown("### Prediction")
    st.success(f"{pred} ({conf:.2f}%)")

    img = load_crop_image(pred)
    if img:
        st.image(img, use_container_width=True)


# ================= COMPARE =================
if compare_clicked:

    arr = np.array(features).reshape(1, -1)

    vqc_pred, vqc_probs = predict_vqc(features, vqc_model, vqc_scaler, vqc_class_names, True)
    vqc_conf = np.max(vqc_probs) * 100

    qnn_pred, qnn_probs = predict_qnn(qnn_model, qnn_scaler, qnn_label_encoder, arr, True)
    qnn_pred = qnn_pred[0]
    qnn_conf = np.max(qnn_probs) * 100

    qknn_pred, qknn_conf = predict_qknn(features, qknn_model, qknn_scaler, qknn_class_names)

    qsvm_pred, qsvm_conf = predict_qsvm(features, qsvm_model, qsvm_scaler, qsvm_classes)

    qre_pred, qre_conf = predict_qreupload(features, qre_model, qre_scaler, qre_classes)

    st.markdown("### Model Comparison")

    st.write(f"VQC → {vqc_pred} ({vqc_conf:.2f}%)")
    st.write(f"QNN → {qnn_pred} ({qnn_conf:.2f}%)")
    st.write(f"QKNN → {qknn_pred} ({qknn_conf:.2f}%)")
    st.write(f"QSVM → {qsvm_pred} ({qsvm_conf:.2f}%)")
    st.write(f"ReUpload → {qre_pred} ({qre_conf:.2f}%)")

    st.bar_chart({
        "VQC": vqc_conf,
        "QNN": qnn_conf,
        "QKNN": qknn_conf,
        "QSVM": qsvm_conf,
        "ReUpload": qre_conf
    })