# ====================================================
# B-cell Epitope Prediction Dashboard
# Stage1 (CatBoost) + Stage2 (XGBoost) Cascade
# ====================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import re
import requests
from streamlit.components.v1 import html

# ====================================================
# 1. Page config
# ====================================================
PROJECT_NAME = "B-cell Epitope Prediction Dashboard"
MODEL_NAME = "2-Stage Cascade Model (High Recall → High Precision)"

st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# 2. Paths
# ====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

CAT_COLS = ["disease", "state", "assay", "method"]

# ====================================================
# 3. Load models
# ====================================================
@st.cache_resource
def load_models():
    model_s1 = joblib.load(os.path.join(MODEL_DIR, "stage1_catboost.pkl"))
    model_s2 = joblib.load(os.path.join(MODEL_DIR, "stage2_xgb.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "encoder_s2.pkl"))
    feature_cols_s1 = joblib.load(
        os.path.join(MODEL_DIR, "stage1_feature_columns.pkl")
    )
    return model_s1, model_s2, encoder, feature_cols_s1

model_s1, model_s2, encoder_s2, feature_cols_s1 = load_models()

# ====================================================
# 4. Utils
# ====================================================
def clean_sequence(seq: str) -> str:
    seq = re.sub(r">.*\n", "", seq)
    seq = re.sub(r"\s+", "", seq)
    return seq.upper()


def sliding_windows(seq, window=16):
    for i in range(len(seq) - window + 1):
        yield i, seq[i:i+window]


def avg_hydrophobicity(seq):
    hydro = {
        "A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,
        "E":-3.5,"G":-0.4,"H":-3.2,"I":4.5,"L":3.8,
        "K":-3.9,"M":1.9,"F":2.8,"P":-1.6,
        "S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2
    }
    return float(np.mean([hydro.get(a, 0) for a in seq]))


def overlap_len(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)

# ====================================================
# 5. Sidebar – Conditions & Thresholds
# ====================================================
with st.sidebar:
    st.header("🔧 Experimental Conditions")

    assay = st.selectbox("Assay", ["antibody binding", "qualitative binding"])
    method = st.selectbox("Method", ["Immunoassay", "high throughput"])
    disease = st.selectbox(
        "Disease", ["disease", "healthy/exposed", "induced/medical", "native", "cancer"]
    )
    state = st.selectbox(
        "State", ["infectious", "cancer", "autoimmune", "allergy", "healthy"]
    )

    st.divider()
    st.header("🎯 Thresholds")

    THRESHOLD_S1 = st.slider(
        "Stage 1 (Recall)", 0.1, 0.9, 0.463, step=0.01
    )

    THRESHOLD_FINAL = st.slider(
        "Final Precision", 0.5, 0.99, 0.957, step=0.01
    )

# ====================================================
# 6. Main UI
# ====================================================
st.title(PROJECT_NAME)
st.caption(MODEL_NAME)

sequence_input = st.text_area(
    "Enter full-length antigen sequence",
    height=180
)

pdb_id = st.text_input(
    "Optional: PDB ID for 3D visualization (e.g. 1A3R)",
    max_chars=4
)

iedb_file = st.file_uploader(
    "Optional: Upload IEDB annotation CSV",
    type=["csv"]
)

run = st.button("Run Epitope Screening", type="primary")

# ====================================================
# 7. Run pipeline
# ====================================================
if run and sequence_input:
    seq = clean_sequence(sequence_input)
    st.success(f"Antigen length: {len(seq)} aa")

    conditions = {
        "assay": assay,
        "method": method,
        "disease": disease,
        "state": state
    }

    results = []

    with st.spinner("Running cascade prediction..."):
        for pos, window in sliding_windows(seq, 16):
            # === embedding lookup assumed ===
            # 여기서는 이미 embedding이 있다고 가정
            # (실제 서비스에서는 embedding 생성/매칭 단계 추가)
            continue

    # ⚠️ 데모용: DB 기반 inference (네 기존 test_metadata 방식)
    df_meta = pd.read_csv(os.path.join(DATA_DIR, "test_metadata.csv"))
    emb = np.load(os.path.join(DATA_DIR, "test_embedding_antigen.npy")).astype("float32")

    df_meta["search_seq"] = df_meta["original_antigen"].apply(
        lambda x: clean_sequence(str(x))
    )

    match = df_meta[df_meta["search_seq"] == seq]

    if match.empty:
        st.error("Sequence not found in embedding database.")
        st.stop()

    idx = match.index[0]
    target_emb = emb[idx]
    meta = match.iloc[0]

    df_emb = pd.DataFrame(
        target_emb.reshape(1, -1),
        columns=[str(i) for i in range(target_emb.shape[0])]
    )

    df_cat = pd.DataFrame([conditions])
    X_s1 = pd.concat([df_emb, df_cat], axis=1)
    X_s1 = X_s1[feature_cols_s1]

    prob_s1 = model_s1.predict_proba(X_s1)[0, 1]

    if prob_s1 < THRESHOLD_S1:
        st.warning("Stage 1 failed.")
        st.stop()

    X_cat = encoder_s2.transform(df_cat)
    X_s2 = np.hstack([target_emb.reshape(1, -1), X_cat])
    prob_s2 = model_s2.predict_proba(X_s2)[0, 1]

    final_score = 0.4 * prob_s1 + 0.6 * prob_s2

    st.subheader("Final Prediction")
    st.metric("Final score", f"{final_score:.4f}")

    if final_score < THRESHOLD_FINAL:
        st.warning("Below final precision threshold.")
        st.stop()

    st.success("✅ Final Positive Epitope Candidate")

    # ====================================================
    # 8. 3D visualization
    # ====================================================
    if pdb_id:
        url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
        r = requests.get(url)
        if r.status_code == 200:
            pdb = r.text.replace("`", "")
            html_code = f"""
            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
            <div id="viewer" style="width:100%; height:500px;"></div>
            <script>
              let v = $3Dmol.createViewer("viewer", {{backgroundColor:"white"}});
              v.addModel(`{pdb}`, "pdb");
              v.setStyle({{}}, {{cartoon:{{color:"lightgray"}}}});
              v.zoomTo(); v.render();
            </script>
            """
            html(html_code, height=520)

    # ====================================================
    # 9. IEDB overlay
    # ====================================================
    if iedb_file:
        iedb = pd.read_csv(iedb_file)
        iedb["Overlap"] = False
        st.subheader("IEDB Annotation Overlay")
        st.dataframe(iedb, use_container_width=True)
