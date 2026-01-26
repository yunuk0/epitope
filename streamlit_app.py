# ====================================================
# B-cell Epitope Prediction Dashboard
# 2-Stage Cascade Model (CatBoost + XGBoost)
# Google Drive large-data loader
# ====================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import re
import requests
import gdown
from streamlit.components.v1 import html

# ====================================================
# 1. Page config (MUST be first)
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
# 4. Google Drive large data loader
# ====================================================
@st.cache_resource
def load_large_data_from_gdrive():
    """
    Download large embedding / metadata files from Google Drive
    (only once, cached)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # 🔽 TODO: 네가 Drive 링크 주면 여기 FILE_ID만 바꾸면 됨
    EMB_FILE_ID = "1wnnErnMNZ87FZLWKyzzv6_fzUNeQetv9"
    META_FILE_ID = "1iAQV2cDfENaEMzcIUq1jiGnI2h8fOa2V"

    emb_path = os.path.join(DATA_DIR, "embedding_antigen.npy")
    meta_path = os.path.join(DATA_DIR, "test_metadata.csv")

    if not os.path.exists(emb_path):
        st.info("⬇️ Downloading embedding data from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={EMB_FILE_ID}",
            emb_path,
            quiet=False
        )

    if not os.path.exists(meta_path):
        st.info("⬇️ Downloading metadata from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={META_FILE_ID}",
            meta_path,
            quiet=False
        )

    # mmap_mode로 메모리 사용 최소화
    emb = np.load(emb_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)

    return emb, meta


# ====================================================
# 5. Utils
# ====================================================
def clean_sequence(seq: str) -> str:
    seq = re.sub(r">.*\n", "", seq)
    seq = re.sub(r"\s+", "", seq)
    return seq.upper()


def avg_hydrophobicity(seq):
    hydro = {
        "A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,
        "Q":-3.5,"E":-3.5,"G":-0.4,"H":-3.2,
        "I":4.5,"L":3.8,"K":-3.9,"M":1.9,
        "F":2.8,"P":-1.6,"S":-0.8,"T":-0.7,
        "W":-0.9,"Y":-1.3,"V":4.2
    }
    return float(np.mean([hydro.get(a, 0) for a in seq]))


# ====================================================
# 6. Sidebar – Conditions & Thresholds
# ====================================================
with st.sidebar:
    st.header("🔧 Experimental Conditions")

    assay = st.selectbox("Assay", ["antibody binding", "qualitative binding"])
    method = st.selectbox("Method", ["Immunoassay", "high throughput"])
    disease = st.selectbox(
        "Disease",
        ["disease", "healthy/exposed", "induced/medical", "native", "cancer"]
    )
    state = st.selectbox(
        "State",
        ["infectious", "cancer", "autoimmune", "allergy", "healthy"]
    )

    st.divider()
    st.header("🎯 Prediction Thresholds")

    THRESHOLD_S1 = st.slider(
        "Stage 1 (Recall threshold)",
        0.1, 0.9, 0.463, step=0.01
    )

    THRESHOLD_FINAL = st.slider(
        "Final Precision threshold",
        0.5, 0.99, 0.957, step=0.01
    )

# ====================================================
# 7. Main UI
# ====================================================
st.title(PROJECT_NAME)
st.caption(MODEL_NAME)

sequence_input = st.text_area(
    "Enter full-length antigen sequence",
    height=180,
    placeholder="Paste full protein sequence here (FASTA header optional)"
)

pdb_id = st.text_input(
    "Optional: PDB ID for 3D structure visualization (e.g. 1A3R)",
    max_chars=4
)

run_btn = st.button("🚀 Run Epitope Prediction", type="primary")

# ====================================================
# 8. Run pipeline
# ====================================================
if run_btn and sequence_input:
    seq = clean_sequence(sequence_input)
    st.success(f"Antigen length: {len(seq)} aa")

    with st.spinner("Loading embedding database..."):
        emb_data, df_meta = load_large_data_from_gdrive()

    # sequence → embedding DB 매칭
    df_meta["search_seq"] = df_meta["original_antigen"].astype(str).apply(clean_sequence)
    match = df_meta[df_meta["search_seq"] == seq]

    if match.empty:
        st.error("❌ Sequence not found in embedding database.")
        st.stop()

    idx = match.index[0]
    target_emb = emb_data[idx]
    meta_row = match.iloc[0]

    # ================= Stage 1 =================
    df_emb = pd.DataFrame(
        target_emb.reshape(1, -1),
        columns=[str(i) for i in range(target_emb.shape[0])]
    )

    df_cat = pd.DataFrame([{
        "assay": assay,
        "method": method,
        "disease": disease,
        "state": state
    }])

    X_s1 = pd.concat([df_emb, df_cat], axis=1)
    X_s1 = X_s1[feature_cols_s1]

    prob_s1 = model_s1.predict_proba(X_s1)[0, 1]

    st.metric("Stage 1 probability", f"{prob_s1:.4f}")

    if prob_s1 < THRESHOLD_S1:
        st.warning("Stage 1 filter failed (below recall threshold).")
        st.stop()

    # ================= Stage 2 =================
    X_cat = encoder_s2.transform(df_cat)
    X_s2 = np.hstack([target_emb.reshape(1, -1), X_cat])

    prob_s2 = model_s2.predict_proba(X_s2)[0, 1]

    final_score = 0.4 * prob_s1 + 0.6 * prob_s2

    st.metric("Final weighted score", f"{final_score:.4f}")

    if final_score < THRESHOLD_FINAL:
        st.warning("Below final precision threshold.")
        st.stop()

    st.success("✅ FINAL POSITIVE epitope candidate")

    # ====================================================
    # 9. 3D Structure Visualization
    # ====================================================
    if pdb_id:
        st.subheader("🧬 3D Structure View")
        url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
        r = requests.get(url)

        if r.status_code == 200:
            pdb_str = r.text.replace("`", "")
            html_code = f"""
            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
            <div id="viewer" style="width:100%; height:500px;"></div>
            <script>
              let viewer = $3Dmol.createViewer("viewer", {{backgroundColor:"white"}});
              viewer.addModel(`{pdb_str}`, "pdb");
              viewer.setStyle({{}}, {{cartoon:{{color:"lightgray"}}}});
              viewer.zoomTo();
              viewer.render();
            </script>
            """
            html(html_code, height=520)
        else:
            st.warning("Failed to load PDB structure.")
