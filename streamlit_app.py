# EpitopeCascade Streamlit Dashboard
# Full-length antigen epitope screening + PDB-based 3D visualization (UI / Dummy inference)

import streamlit as st
import numpy as np
import pandas as pd
import re
from typing import List
import requests
import py3Dmol
from streamlit.components.v1 import html

# =============================
# Project / Model naming
# =============================
PROJECT_NAME = "EpitopeCascade"
MODEL_NAME = "Condition-aware Ensemble Cascade Model (UI / Dummy mode)"

st.set_page_config(page_title=PROJECT_NAME, layout="wide")

# ======================================================
# Utility functions
# ======================================================

def clean_sequence(seq: str) -> str:
    seq = seq.strip()
    seq = re.sub(r">.*\n", "", seq)
    seq = re.sub(r"\s+", "", seq)
    return seq.upper()


def generate_epitopes(window: str, min_len=8, max_len=16):
    epis = []
    for l in range(min_len, max_len + 1):
        for i in range(len(window) - l + 1):
            epis.append((window[i:i+l], i, i+l))
    return epis


# ======================================================
# Dummy prediction logic
# ======================================================

def dummy_predict_score(epitope: str, conditions: dict) -> float:
    base = sum(ord(a) for a in epitope) % 100 / 100
    if conditions["Disease"] == "cancer":
        base -= 0.05
    if conditions["Assay"] == "antibody binding":
        base += 0.05
    return max(0.0, min(1.0, base))


# ======================================================
# Amino acid properties
# ======================================================

hydrophobicity = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}


def avg_hydrophobicity(seq):
    return float(np.mean([hydrophobicity.get(a, 0) for a in seq]))


# ======================================================
# Sidebar – Experimental conditions
# ======================================================

st.sidebar.title("Experimental Conditions")

assay = st.sidebar.selectbox("Assay", ["antibody binding", "qualitative binding"])
method = st.sidebar.selectbox("Method", ["Immunoassay", "high throughput"])
disease = st.sidebar.selectbox(
    "Disease",
    ["disease", "healthy/exposed", "induced/medical", "native", "cancer"]
)
state = st.sidebar.selectbox(
    "State",
    ["infectious", "cancer", "autoimmune", "allergy", "healthy"]
)

conditions = {
    "Assay": assay,
    "Method": method,
    "Disease": disease,
    "State": state
}

# =============================
# Model information (display only)
# =============================
with st.sidebar.expander("Model information", expanded=False):
    st.markdown("**Ensemble Cascade Model (fixed parameters)**")
    st.slider("Recall threshold", 0.0, 1.0, 0.85, step=0.01, disabled=True)
    st.slider("Precision threshold", 0.0, 1.0, 0.80, step=0.01, disabled=True)
    st.slider("Alpha (ensemble weight)", 0.0, 1.0, 0.60, step=0.05, disabled=True)


# ======================================================
# Main UI
# ======================================================

st.title(PROJECT_NAME)
st.caption(MODEL_NAME)

sequence_input = st.text_area(
    "Enter full-length antigen sequence",
    height=220
)

pdb_id = st.text_input(
    "Optional: Enter PDB ID for 3D structure visualization (e.g. 1A3R)",
    max_chars=4
)

run = st.button("Run Epitope Screening")


# ======================================================
# Analysis pipeline
# ======================================================

if run and sequence_input:
    seq = clean_sequence(sequence_input)
    st.success(f"Antigen length: {len(seq)} aa")

    records = []
    window_size = 16

    with st.spinner("Running sliding-window epitope screening..."):
        for w_start in range(len(seq) - window_size + 1):
            window = seq[w_start:w_start + window_size]
            for ep, s, e in generate_epitopes(window):
                score = dummy_predict_score(ep, conditions)
                records.append({
                    "Epitope": ep,
                    "Score": score,
                    "Epitope_start": w_start + s + 1,
                    "Epitope_end": w_start + e,
                    "Length": len(ep),
                    "Avg_hydrophobicity": avg_hydrophobicity(ep)
                })

    df = pd.DataFrame(records).sort_values("Score", ascending=False)
    top = df.iloc[0]

    # =============================
    # Top epitope summary
    # =============================

    st.subheader("Top-ranked Epitope Candidate")
    c1, c2, c3 = st.columns(3)
    c1.metric("Epitope", top.Epitope)
    c2.metric("Score", f"{top.Score:.3f}")
    c3.metric("AA position", f"{int(top.Epitope_start)} – {int(top.Epitope_end)}")

    st.markdown(f"**Average hydrophobicity:** {top.Avg_hydrophobicity:.2f}")

    st.subheader("Top candidate epitopes")
    st.dataframe(df.head(20), use_container_width=True)

    # =============================
    # 3D Structure visualization
    # =============================

    if pdb_id:
        st.subheader("3D Structure View (Top Epitope Highlighted)")
        pdb_id = pdb_id.lower()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        r = requests.get(url)

        if r.status_code == 200:
            pdb_str = r.text
            view = py3Dmol.view(width=700, height=500)
            view.addModel(pdb_str, 'pdb')
            view.setStyle({'cartoon': {'color': 'lightgray'}})

            view.setStyle(
                {'resi': list(range(int(top.Epitope_start), int(top.Epitope_end) + 1))},
                {'stick': {'color': 'red'}}
            )

            view.zoomTo()
            html(view._make_html(), height=520)
        else:
            st.warning("Failed to load PDB structure. Please check the PDB ID.")
