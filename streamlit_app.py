# VaxOptiML Streamlit Dashboard
# Full-length antigen epitope screening with conditional prediction

import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
from typing import List

PROJECT_NAME = "EpitopeCascade"
MODEL_NAME = "Condition-aware Ensemble Cascade Model"

st.set_page_config(page_title=PROJECT_NAME, layout="wide")

# ======================================================
# Utility functions
# ======================================================

def clean_sequence(seq: str) -> str:
    seq = seq.strip()
    seq = re.sub(r">.*\n", "", seq)  # remove FASTA headers
    seq = re.sub(r"\s+", "", seq)     # remove spaces/newlines
    return seq.upper()


def sliding_window(seq: str, k: int) -> List[str]:
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def generate_epitopes(window: str, min_len=8, max_len=16):
    epis = []
    for l in range(min_len, max_len + 1):
        for i in range(len(window) - l + 1):
            epis.append((window[i:i+l], i, i+l))
    return epis


# ======================================================
# Dummy model loader (replace with real pickle)
# ======================================================

@st.cache_resource
def load_model():
    # with open("epitope_model.pkl", "rb") as f:
    #     model = pickle.load(f)
    model = None  # placeholder
    return model


def predict_score(epitope: str, conditions: dict) -> float:
    # Placeholder logic – replace with model.predict
    base = sum([ord(a) for a in epitope]) % 100 / 100
    weight = 0
    if conditions["Disease"] == "cancer":
        weight -= 0.05
    if conditions["Assay"] == "antibody binding":
        weight += 0.05
    return max(0, min(1, base + weight))


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
    return np.mean([hydrophobicity.get(a, 0) for a in seq])


# ======================================================
# Sidebar – Experimental conditions
# ======================================================

st.sidebar.title("Experimental Conditions")

# =============================
# Model information (bottom-left)
# =============================
with st.sidebar.expander("Model information", expanded=False):
    st.markdown("**Ensemble Cascade Model**")
    recall_threshold = st.slider(
        "Recall threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        help="Recall cutoff used in stage-1 screening"
    )
    precision_threshold = st.slider(
        "Precision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.80,
        step=0.01,
        help="Precision cutoff used in final epitope selection"
    )
    alpha = st.slider(
        "Alpha (ensemble weight)",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="Weighting factor between cascade stages"
    )

    st.caption("⚠ Parameters are displayed for transparency. Actual values are fixed in the trained ensemble cascade model.")

assay = st.sidebar.selectbox(
    "Assay",
    ["antibody binding", "qualitative binding"]
)

method = st.sidebar.selectbox(
    "Method",
    ["Immunoassay", "high throughput"]
)

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


# ======================================================
# Main UI
# ======================================================

st.title(PROJECT_NAME)
st.caption(MODEL_NAME)

sequence_input = st.text_area(
    "Enter full-length antigen sequence (no length limit)",
    height=250
)

run = st.button("Run Epitope Screening")


# ======================================================
# Analysis pipeline
# ======================================================

if run and sequence_input:
    seq = clean_sequence(sequence_input)
    st.success(f"Antigen length: {len(seq)} aa")

    model = load_model()

    records = []
    window_size = 16

    with st.spinner("Running sliding-window epitope screening..."):
        for w_start in range(len(seq) - window_size + 1):
            window = seq[w_start:w_start + window_size]
            epitopes = generate_epitopes(window)

            for ep, s, e in epitopes:
                score = predict_score(ep, conditions)
                records.append({
                    "Epitope": ep,
                    "Score": score,
                    "Window_start": w_start + 1,
                    "Epitope_start": w_start + s + 1,
                    "Epitope_end": w_start + e,
                    "Length": len(ep),
                    "Avg_hydrophobicity": avg_hydrophobicity(ep)
                })

    df = pd.DataFrame(records).sort_values("Score", ascending=False)

    # ======================================================
    # Top epitope summary
    # ======================================================

    top = df.iloc[0]

    st.subheader("Top-ranked Epitope")

    col1, col2, col3 = st.columns(3)
    col1.metric("Epitope sequence", top.Epitope)
    col2.metric("Prediction score", f"{top.Score:.3f}")
    col3.metric("AA position", f"{int(top.Epitope_start)} – {int(top.Epitope_end)}")

    st.markdown(f"**Average hydrophobicity:** {top.Avg_hydrophobicity:.2f}")

    # ======================================================
    # Table view
    # ======================================================

    st.subheader("Top candidate epitopes")
    st.dataframe(df.head(50), use_container_width=True)

    # ======================================================
    # Antigen highlight visualization
    # ======================================================

    st.subheader("Antigen sequence (top epitope highlighted)")

    highlight = {i: 0 for i in range(len(seq))}
    for i in range(int(top.Epitope_start) - 1, int(top.Epitope_end)):
        highlight[i] = 1

    def render_seq():
        html = ""
        for i, aa in enumerate(seq):
            if highlight[i]:
                html += f"<span style='background-color: rgba(255,0,0,0.6); font-family:monospace'>{aa}</span>"
            else:
                html += f"<span style='font-family:monospace'>{aa}</span>"
        return html

    st.markdown(render_seq(), unsafe_allow_html=True)
