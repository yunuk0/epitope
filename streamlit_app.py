# ====================================================
# B-cell Epitope Prediction Dashboard (Inference Only)
# Upload embedding_antigen.npy → Stage1 → Stage2 → Ranking
# ====================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ====================================================
# Page config (MUST be first Streamlit call)
# ====================================================
st.set_page_config(
    page_title="B-cell Epitope Prediction Dashboard",
    layout="wide",
    page_icon=":dna:",
)

# ====================================================
# Constants (default values from your training)
# ====================================================
DEFAULT_THRESHOLD_FINAL = 0.957
W1, W2 = 0.4, 0.6
CAT_COLS = ["disease", "state", "assay", "method"]

# ====================================================
# Load models (no feature_columns.pkl required)
# ====================================================
@st.cache_resource

def load_models():
    try:
        model_s1 = joblib.load("stage1_catboost.pkl")
        model_s2 = joblib.load("stage2_xgb.pkl")
        encoder_s2 = joblib.load("encoder_s2.pkl")
        return model_s1, model_s2, encoder_s2
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model_s1, model_s2, encoder_s2 = load_models()

# ====================================================
# Sidebar – controls
# ====================================================
st.sidebar.title("Inference settings")
final_threshold = st.sidebar.slider(
    "Final precision threshold",
    min_value=0.70,
    max_value=0.957,
    value=DEFAULT_THRESHOLD_FINAL,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model info**")
st.sidebar.markdown("• Stage 1: CatBoost (High Recall)")
st.sidebar.markdown("• Stage 2: XGBoost (High Precision)")
st.sidebar.markdown("• Final score = 0.4 × S1 + 0.6 × S2")

# ====================================================
# Main UI
# ====================================================
st.title(":dna: B-cell Epitope Prediction Dashboard")
st.caption("2-Stage Cascade Model | Embedding-based Inference")

st.markdown(
    """
**Input requirement**  
• Upload **ProtT5-generated antigen embeddings** (`embedding_antigen.npy`)  
• Shape must be **(N, 1024)** where each row corresponds to one 16-mer antigen window
"""
)

uploaded_emb = st.file_uploader(
    "Upload embedding_antigen.npy",
    type=["npy"],
)

# ====================================================
# Inference pipeline
# ====================================================
if uploaded_emb is not None:
    try:
        X_emb = np.load(uploaded_emb).astype("float32")
    except Exception as e:
        st.error(f"Failed to load embedding file: {e}")
        st.stop()

    if X_emb.ndim != 2 or X_emb.shape[1] != 1024:
        st.error("Embedding must have shape (N, 1024)")
        st.stop()

    st.success(f"Loaded embedding matrix: {X_emb.shape}")

    # ----------------------------------------------------
    # Metadata input (global conditions)
    # ----------------------------------------------------
    st.subheader("Experimental conditions")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        disease = st.selectbox("Disease", ["disease", "healthy/exposed", "induced/medical", "native", "cancer"])
    with col2:
        state = st.selectbox("State", ["infectious", "cancer", "autoimmune", "allergy", "healthy"])
    with col3:
        assay = st.selectbox("Assay", ["antibody binding", "qualitative binding"])
    with col4:
        method = st.selectbox("Method", ["Immunoassay", "high throughput"])

    df_meta = pd.DataFrame({
        "disease": [disease] * len(X_emb),
        "state": [state] * len(X_emb),
        "assay": [assay] * len(X_emb),
        "method": [method] * len(X_emb),
    })

    # ----------------------------------------------------
    # Stage 1
    # ----------------------------------------------------
    df_emb = pd.DataFrame(X_emb, columns=[str(i) for i in range(1024)])
    X_s1 = pd.concat([df_emb, df_meta[CAT_COLS]], axis=1)

    with st.spinner("Running Stage 1 (CatBoost)…"):
        prob_s1 = model_s1.predict_proba(X_s1)[:, 1]

    pass_mask = prob_s1 >= 0.0  # no hard cutoff, pass all forward

    # ----------------------------------------------------
    # Stage 2
    # ----------------------------------------------------
    with st.spinner("Running Stage 2 (XGBoost)…"):
        X_cat_s2 = encoder_s2.transform(df_meta[CAT_COLS])
        X_s2 = np.hstack([X_emb, X_cat_s2])
        prob_s2 = model_s2.predict_proba(X_s2)[:, 1]

    # ----------------------------------------------------
    # Final score
    # ----------------------------------------------------
    final_score = (W1 * prob_s1) + (W2 * prob_s2)

    df_result = pd.DataFrame({
        "Index": np.arange(len(X_emb)),
        "Stage1_Prob": prob_s1,
        "Stage2_Prob": prob_s2,
        "Final_Score": final_score,
        "Final_Pass": final_score >= final_threshold,
    }).sort_values("Final_Score", ascending=False)

    # ----------------------------------------------------
    # Results display
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader(":trophy: Top-ranked Epitope Candidates")

    st.dataframe(df_result.head(20), use_container_width=True)

    passed = df_result[df_result["Final_Pass"]]
    st.success(f"Final positives (threshold {final_threshold:.3f}): {len(passed)} / {len(df_result)}")

    # ----------------------------------------------------
    # Download
    # ----------------------------------------------------
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full prediction table",
        data=csv,
        file_name="epitope_predictions.csv",
        mime="text/csv",
    )

else:
    st.info("Please upload an embedding_antigen.npy file to begin inference.")
