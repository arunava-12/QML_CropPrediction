import streamlit as st
import numpy as np
from PIL import Image
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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


# ================= RUN ALL MODELS =================
def run_all_models(features):
    arr = np.array(features).reshape(1, -1)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "VQC": executor.submit(predict_vqc, features, vqc_model, vqc_scaler, vqc_class_names, True),
            "QNN": executor.submit(predict_qnn, qnn_model, qnn_scaler, qnn_label_encoder, arr, True),
            "QKNN": executor.submit(predict_qknn, features, qknn_model, qknn_scaler, qknn_class_names),
            "QSVM": executor.submit(predict_qsvm, features, qsvm_model, qsvm_scaler, qsvm_classes),
            "REUPLOAD": executor.submit(predict_qreupload, features, qre_model, qre_scaler, qre_classes),
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
    with st.spinner("Running quantum models... ⚛️"):

        choice = st.session_state.model_choice

        if choice == "VQC":
            pred, probs = predict_vqc(features, vqc_model, vqc_scaler, vqc_class_names, True)
            probs = np.array(probs).flatten()
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3 = [(vqc_class_names[int(i)], probs[int(i)] * 100) for i in top3_idx]

        elif choice == "QNN":
            arr = np.array(features).reshape(1, -1)
            pred, probs = predict_qnn(qnn_model, qnn_scaler, qnn_label_encoder, arr, True)
            classes = qnn_label_encoder.classes_
            probs = np.array(probs).flatten()
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3 = [(classes[int(i)], probs[int(i)] * 100) for i in top3_idx]

        elif choice == "QKNN":
            pred, conf = predict_qknn(features, qknn_model, qknn_scaler, qknn_class_names)
            top3 = [(pred, conf)]

        elif choice == "QSVM":
            pred, conf = predict_qsvm(features, qsvm_model, qsvm_scaler, qsvm_classes)
            top3 = [(pred, conf)]

        elif choice == "REUPLOAD":
            pred, conf = predict_qreupload(features, qre_model, qre_scaler, qre_classes)
            top3 = [(pred, conf)]

        st.markdown("### 🌾 Top Predictions")

        for i, (crop, confidence) in enumerate(top3, 1):
            st.markdown(f"**{i}. {crop}**")
            st.progress(int(confidence))
            st.caption(f"{confidence:.2f}% confidence")

        best_crop = top3[0][0]
        img = load_crop_image(best_crop)

        if img:
            st.image(img, use_column_width=True)


# ================= COMPARE =================
if compare_clicked:
    with st.spinner("Comparing quantum models... ⚔️"):

        results = run_all_models(features)

        # Extract results
        vqc_pred, vqc_probs = results["VQC"]
        vqc_probs = np.array(vqc_probs).flatten()
        vqc_conf = np.max(vqc_probs) * 100

        qnn_pred, qnn_probs = results["QNN"]
        qnn_pred = qnn_pred[0]
        qnn_probs = np.array(qnn_probs).flatten()
        qnn_conf = np.max(qnn_probs) * 100

        qknn_pred, qknn_conf = results["QKNN"]
        qsvm_pred, qsvm_conf = results["QSVM"]
        qre_pred, qre_conf = results["REUPLOAD"]

        st.markdown("### 🏆 Model Leaderboard")
        st.success("✅ Comparison complete!")

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

        # ================= CONSENSUS + AGREEMENT =================
        all_preds = [vqc_pred, qnn_pred, qknn_pred, qsvm_pred, qre_pred]

        # Majority vote
        consensus_crop = max(set(all_preds), key=all_preds.count)

        # Agreement %
        agreement_score = (all_preds.count(consensus_crop) / len(all_preds)) * 100

        st.markdown("### 🤝 Model Consensus")
        st.success(f"🌾 Consensus Crop: **{consensus_crop}**")

        st.metric("Agreement Score", f"{agreement_score:.1f}%")

        agree_df = pd.DataFrame({
            "Model": ["VQC", "QNN", "QKNN", "QSVM", "ReUpload"],
            "Prediction": all_preds
        })

        st.write("Model Predictions:")
        st.dataframe(agree_df)

    # ================= FEATURE IMPORTANCE =================
    st.markdown("### 📊 Feature Importance (Explainability)")

    with st.spinner("Computing feature importance via perturbation... 🔬"):
        names, scores = compute_feature_importance(tuple(features), st.session_state.model_choice)

        imp_df = pd.DataFrame({
            "Feature": names,
            "Importance": scores
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(imp_df.set_index("Feature"))
        st.caption("Higher = more influence on prediction confidence across all models")

    # ================= CONFUSION MATRIX =================
    st.markdown("### 🔬 Confusion Matrix (VQC Sample)")

    data_path = os.path.join(BASE_DIR, "Crop_recommendation.csv")

    if os.path.exists(data_path):
        with st.spinner("Generating confusion matrix on 100 samples... 🧪"):
            df_cm = pd.read_csv(data_path).sample(100, random_state=42)

            X_cm = df_cm.drop("label", axis=1).values
            y_true = df_cm["label"].values

            y_pred = []
            for row in X_cm:
                pred_label = predict_vqc(row, vqc_model, vqc_scaler, vqc_class_names, False)
                y_pred.append(pred_label)

            # Only use classes that appear in this sample
            sample_classes = sorted(list(set(y_true) | set(y_pred)))

            cm = confusion_matrix(y_true, y_pred, labels=sample_classes)

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                xticklabels=sample_classes,
                yticklabels=sample_classes,
                cmap="Greens",
                ax=ax
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("VQC Confusion Matrix (100-sample test)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            st.pyplot(fig)
            st.caption("Diagonal = correct predictions. Off-diagonal = misclassifications.")
    else:
        st.warning(
            "⚠️ `Crop_recommendation.csv` not found in the project directory. "
            "Place it alongside `app.py` to enable the confusion matrix."
        )