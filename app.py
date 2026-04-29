import streamlit as st
import numpy as np
from PIL import Image
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from concurrent.futures import ThreadPoolExecutor

from vqc.vqc_model import load_vqc_model, predict_vqc
from qnn.qnn_model import load_qnn_model, predict_qnn
from qknn.qknn_model import load_qknn_model, predict_qknn
from qsvm.qsvm_model import load_qsvm_model, predict_qsvm
from qreupload.qreupload_model import load_qreupload_model, predict_qreupload


# ================= BASE PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


# ================= SAFE LOAD =================
try:
    (
        vqc_model, vqc_scaler, vqc_class_names,
        qnn_model, qnn_scaler, qnn_label_encoder,
        qknn_model, qknn_scaler, qknn_class_names,
        qsvm_model, qsvm_scaler, qsvm_classes,
        qre_model, qre_scaler, qre_classes
    ) = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


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
rainfall = col2.number_input("Rainfall", value=200.0)

features = [N, P, K, temperature, humidity, ph, rainfall]


# ================= IMAGE =================
def load_crop_image(crop):
    path = os.path.join(BASE_DIR, "assets", f"{crop.lower()}.jpg")
    return Image.open(path) if os.path.exists(path) else None


def timed_predict(name, func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start


# ================= RUN ALL MODELS =================
def run_all_models(features):
    arr = np.array(features).reshape(1, -1)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            "VQC": executor.submit(timed_predict, "VQC", predict_vqc, features, vqc_model, vqc_scaler, vqc_class_names, True),
            "QNN": executor.submit(timed_predict, "QNN", predict_qnn, qnn_model, qnn_scaler, qnn_label_encoder, arr, True),
            "QKNN": executor.submit(timed_predict, "QKNN", predict_qknn, features, qknn_model, qknn_scaler, qknn_class_names),
            "QSVM": executor.submit(timed_predict, "QSVM", predict_qsvm, features, qsvm_model, qsvm_scaler, qsvm_classes),
            "REUPLOAD": executor.submit(timed_predict, "REUPLOAD", predict_qreupload, features, qre_model, qre_scaler, qre_classes),
        }

        results = {k: f.result() for k, f in futures.items()}

    return results


# ================= FEATURE IMPORTANCE =================
@st.cache_data(show_spinner=False)
def compute_feature_importance(features_tuple, model_choice):
    features = list(features_tuple)
    feature_names = ["N", "P", "K", "Temp", "Humidity", "pH", "Rain"]

    # ✅ Use only VQC for fast perturbation-based importance
    def get_conf(f):
        pred, probs = predict_vqc(f, vqc_model, vqc_scaler, vqc_class_names, True)
        return np.max(np.array(probs).flatten())

    base_conf = get_conf(features)

    importances = []
    for i in range(len(features)):
        temp_features = features.copy()
        temp_features[i] *= 0.9  # perturb by 10%
        new_conf = get_conf(temp_features)
        importances.append(abs(base_conf - new_conf))

    return feature_names, importances

# ================= MAIN BUTTONS =================
colA, colB = st.columns(2)
predict_clicked = colA.button("Predict Crop 🚀")
compare_clicked = colB.button("Compare All Models ⚔️")


# ================= PREDICT =================
if predict_clicked:
    start_time = time.time()
    with st.spinner(f"Running {st.session_state.model_choice} model... ⚛️"):

        choice = st.session_state.model_choice

        if choice == "VQC":
            pred, probs = predict_vqc(features, vqc_model, vqc_scaler, vqc_class_names, True)
            probs = np.array(probs).flatten()
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3 = [(vqc_class_names[int(i)], probs[int(i)] * 100) for i in top3_idx]
            model_info = "Variational Quantum Classifier (VQC) uses a parameterised quantum circuit to map features into a high-dimensional Hilbert space for classification."

        elif choice == "QNN":
            arr = np.array(features).reshape(1, -1)
            pred, probs = predict_qnn(qnn_model, qnn_scaler, qnn_label_encoder, arr, True)
            classes = qnn_label_encoder.classes_
            probs = np.array(probs).flatten()
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3 = [(classes[int(i)], probs[int(i)] * 100) for i in top3_idx]
            model_info = "Quantum Neural Network (QNN) combines quantum layers with classical layers, utilizing quantum entanglement to learn complex crop patterns."

        elif choice == "QKNN":
            pred, conf = predict_qknn(features, qknn_model, qknn_scaler, qknn_class_names)
            top3 = [(pred, conf)]
            model_info = "Quantum K-Nearest Neighbors (QKNN) uses the Swap Test to calculate quantum fidelity between your soil data and the training set."

        elif choice == "QSVM":
            pred, conf = predict_qsvm(features, qsvm_model, qsvm_scaler, qsvm_classes)
            top3 = [(pred, conf)]
            model_info = "Quantum Support Vector Machine (QSVM) uses a quantum-enhanced kernel to find the optimal boundary between different crop types."

        elif choice == "REUPLOAD":
            pred, conf = predict_qreupload(features, qre_model, qre_scaler, qre_classes)
            top3 = [(pred, conf)]
            model_info = "Data Re-Uploading QNN increases the 'expressivity' of the quantum circuit by passing the same input features through multiple layers."

        end_time = time.time()
        elapsed = end_time - start_time

        # UI Layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🌾 Top Predictions")
            for i, (crop, confidence) in enumerate(top3, 1):
                st.markdown(f"**{i}. {crop}**")
                st.progress(int(confidence))
                st.caption(f"{confidence:.2f}% confidence")
            
            best_crop = top3[0][0]
            img = load_crop_image(best_crop)
            if img:
                st.image(img, use_column_width=True)

        with col2:
            st.markdown("### ⚛️ Quantum Insights")
            st.metric("Inference Time", f"{elapsed:.4f} sec")
            st.write(f"**Model Type:** {choice}")
            st.write(model_info)
            
            if top3[0][1] > 80:
                st.success("High Confidence Prediction")
            elif top3[0][1] > 50:
                st.warning("Moderate Confidence Prediction")
            else:
                st.error("Low Confidence Prediction")

        # ================= NEW: SOIL SUITABILITY & RECOMMENDATIONS =================
        st.markdown("---")
        st.markdown(f"### 📊 Soil Suitability Analysis for **{best_crop.capitalize()}**")
        
        # Load data to get averages
        df_stats = pd.read_csv(os.path.join(BASE_DIR, "Crop_recommendation.csv"))
        crop_stats = df_stats[df_stats['label'] == best_crop.lower()].mean(numeric_only=True)
        
        feature_names = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"]
        input_values = [N, P, K, temperature, humidity, ph, rainfall]
        ideal_values = [crop_stats['N'], crop_stats['P'], crop_stats['K'], crop_stats['temperature'], 
                        crop_stats['humidity'], crop_stats['ph'], crop_stats['rainfall']]

        # Display comparison
        comp_data = pd.DataFrame({
            "Feature": feature_names,
            "Your Value": input_values,
            "Ideal (Avg)": ideal_values
        })
        
        col_s1, col_s2 = st.columns([2, 1])
        
        with col_s1:
            st.dataframe(comp_data.set_index("Feature"), use_container_width=True)
        
        with col_s2:
            st.markdown("#### 💡 Smart Tips")
            tips = []
            if N < ideal_values[0] * 0.8: tips.append("Add Nitrogen-rich fertilizer (Urea).")
            if P < ideal_values[1] * 0.8: tips.append("Increase Phosphorus (DAP).")
            if K < ideal_values[2] * 0.8: tips.append("Boost Potassium levels.")
            if ph < 6.0: tips.append("Soil is acidic; consider adding lime.")
            if ph > 7.5: tips.append("Soil is alkaline; consider adding sulfur.")
            if rainfall < ideal_values[6] * 0.7: tips.append("Ensure regular irrigation.")
            
            if not tips:
                st.write("✅ Your soil is nearly ideal for this crop!")
            else:
                for tip in tips:
                    st.write(f"- {tip}")


# ================= COMPARE =================
if compare_clicked:
    with st.spinner("Comparing quantum models... ⚔️"):

        results = run_all_models(features)

        # Extract results and times
        (vqc_res, vqc_probs), vqc_time = results["VQC"]
        vqc_pred = vqc_res
        vqc_probs = np.array(vqc_probs).flatten()
        vqc_conf = np.max(vqc_probs) * 100

        (qnn_res, qnn_probs), qnn_time = results["QNN"]
        qnn_pred = qnn_res[0]
        qnn_probs = np.array(qnn_probs).flatten()
        qnn_conf = np.max(qnn_probs) * 100

        (qknn_pred, qknn_conf), qknn_time = results["QKNN"]
        (qsvm_pred, qsvm_conf), qsvm_time = results["QSVM"]
        (qre_pred, qre_conf), qre_time = results["REUPLOAD"]

        st.markdown("### 🤝 Model Consensus")
        
        leaderboard = [
            ("VQC", vqc_pred, vqc_conf, vqc_time),
            ("QNN", qnn_pred, qnn_conf, qnn_time),
            ("QKNN", qknn_pred, qknn_conf, qknn_time),
            ("QSVM", qsvm_pred, qsvm_conf, qsvm_time),
            ("ReUpload", qre_pred, qre_conf, qre_time),
        ]

        leaderboard = sorted(leaderboard, key=lambda x: x[2], reverse=True)

        # ================= CONSENSUS =================
        preds = [m[1] for m in leaderboard]
        from collections import Counter
        counts = Counter(preds)
        most_common_crop, count = counts.most_common(1)[0]
        consensus_pct = (count / len(preds)) * 100

        col1, col2 = st.columns(2)
        col1.metric("Final Recommendation", most_common_crop)
        col2.metric("Consensus Strength", f"{count}/{len(preds)} Models", f"{consensus_pct:.0f}% Agreement")

        if consensus_pct > 60:
            st.success(f"High confidence: {count} models agree on **{most_common_crop}**.")
        else:
            st.warning("Low consensus: Models are divided. Consider the top-ranked model (VQC/QNN) for better reliability.")

        # ================= LEADERBOARD =================
        st.markdown("### 🏆 Model Leaderboard")
        
        df_display = pd.DataFrame(leaderboard, columns=["Model", "Prediction", "Confidence (%)", "Time (s)"])
        st.table(df_display)

        # ================= CHARTS =================
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("**Confidence Comparison**")
            st.bar_chart(df_display.set_index("Model")["Confidence (%)"])
            
        with col_c2:
            st.markdown("**Inference Latency (Seconds)**")
            st.area_chart(df_display.set_index("Model")["Time (s)"])

        leaderboard = [
            ("VQC", vqc_pred, vqc_conf),
            ("QNN", qnn_pred, qnn_conf),
            ("QKNN", qknn_pred, qknn_conf),
            ("QSVM", qsvm_pred, qsvm_conf),
            ("ReUpload", qre_pred, qre_conf),
        ]

        leaderboard = sorted(leaderboard, key=lambda x: x[2], reverse=True)

        for rank, (model, pred, conf) in enumerate(leaderboard, 1):
            st.write(f"{rank}. {model} → {pred} ({conf:.2f}%)")

        df = pd.DataFrame({
            "Model": [m[0] for m in leaderboard],
            "Confidence": [m[2] for m in leaderboard]
        })

        st.bar_chart(df.set_index("Model"))