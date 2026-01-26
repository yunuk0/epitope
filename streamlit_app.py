import streamlit as st
import numpy as np
import pandas as pd
import os

from utils.model_loader import load_stage1_model, load_stage2
from utils.iedb import load_iedb, annotate_overlap
from utils.visualization import render_3dmol

# ===============================
st.set_page_config(page_title="Epitope Dashboard", layout="wide")

MODEL_DIR = "models"
DATA_DIR = "data"

# ===============================
st.sidebar.header("Model Settings")
threshold = st.sidebar.selectbox("Stage1 Threshold", [0.2, 0.3, 0.4])
overlay_on = st.sidebar.toggle("IEDB Overlay", value=True)

# ===============================
@st.cache_resource
def load_base_data():
    meta = pd.read_csv(os.path.join(DATA_DIR, "test_metadata.csv"))
    emb = np.load(os.path.join(DATA_DIR, "test_embedding_antigen.npy"))
    return meta, emb

df_meta, emb_data = load_base_data()

model_s1 = load_stage1_model(threshold, MODEL_DIR)
model_s2, enc_s2 = load_stage2(MODEL_DIR)

iedb_df = load_iedb(os.path.join(DATA_DIR, "iedb_epitope.csv")) if overlay_on else None

# ===============================
st.title("🧬 B-cell Epitope Prediction")

seq = st.text_area("Antigen Sequence")

if st.button("Run"):
    match = df_meta[df_meta.antigen == seq]

    if match.empty:
        st.error("Sequence not found")
    else:
        idx = match.index[0]
        emb = emb_data[idx]

        X = pd.DataFrame(emb.reshape(1,-1))
        prob1 = model_s1.predict_proba(X)[0,1]

        if prob1 < threshold:
            st.warning("Stage1 fail")
        else:
            st.success(f"Stage1 pass ({prob1:.3f})")

            # 예시 epitope 위치
            pred_range = (50, 65)

            overlaps = []
            if overlay_on:
                overlaps = annotate_overlap(pred_range, iedb_df)
                st.dataframe(pd.DataFrame(overlaps))

            # pdb_text는 AlphaFold fetch 결과라고 가정
            pdb_text = open("example.pdb").read()
            render_3dmol(
                pdb_text,
                pred_range,
                [(row.start, row.end) for _, row in iedb_df.iterrows()] if overlay_on else None
            )
