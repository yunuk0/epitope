# ============================================================
# EpitopeCascade Streamlit Dashboard
# Full-length antigen epitope screening + PDB 3D visualization
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import re
import requests
from streamlit.components.v1 import html

# =============================
# Project / Model naming
# =============================
PROJECT_NAME = "EpitopeCascade"
MODEL_NAME = "Condition-aware Ensemble Cascade Model (UI / Dummy inference)"

st.set_page_config(page_title=PROJECT_NAME, layout="wide")

# =============================
# Utility functions
# =============================

def clean_sequence(seq: str) -> str:
    seq = seq.strip()
    seq = re.sub(r">.*\n", "", seq)
    seq = re.sub(r"\s+", "", seq)
    return seq.upper()


def generate_epitopes(window: str, min_len=8, max_len=16):
    out = []
    for L in range(min_len, max_len + 1):
        for i in range(len(window) - L + 1):
            out.append((window[i:i+L], i, i+L))
    return out


# =============================
# Dummy prediction (placeholder)
# =============================

def dummy_predict_score(epitope: str, conditions: dict) -> float:
    score = sum(ord(a) for a in epitope) % 100 / 100
    if conditions["Disease"] == "cancer":
        score -= 0.05
    if conditions["Assay"] == "antibody binding":
        score += 0.05
    return max(0.0, min(1.0, score))


# =============================
# Amino acid properties
# =============================

hydrophobicity = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}

def avg_hydro(seq):
    return float(np.mean([hydrophobicity.get(a, 0) for a in seq]))


# =============================
# Sidebar – Experimental conditions
# =============================

st.sidebar.title("Experimental Conditions")

conditions = {
    "Assay": st.sidebar.selectbox("Assay", ["antibody binding", "qualitative binding"]),
    "Method": st.sidebar.selectbox("Method", ["Immunoassay", "high throughput"]),
    "Disease": st.sidebar.selectbox(
        "Disease",
        ["disease", "healthy/exposed", "induced/medical", "native", "cancer"]
    ),
    "State": st.sidebar.selectbox(
        "State",
        ["infectious", "cancer", "autoimmune", "allergy", "healthy"]
    )
}

with st.sidebar.expander("Model information"):
    st.slider("Recall threshold", 0.0, 1.0, 0.85, disabled=True)
    st.slider("Precision threshold", 0.0, 1.0, 0.80, disabled=True)
    st.slider("Alpha (ensemble weight)", 0.0, 1.0, 0.60, disabled=True)


# =============================
# Main UI
# =============================

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


# =============================
# Analysis
# =============================

if run and sequence_input:

    seq = clean_sequence(sequence_input)
    st.success(f"Antigen length: {len(seq)} aa")

    records = []
    window_size = 16

    with st.spinner("Running sliding-window epitope screening..."):
        for w in range(len(seq) - window_size + 1):
            window = seq[w:w + window_size]
            for ep, s, e in generate_epitopes(window):
                records.append({
                    "Epitope": ep,
                    "Score": dummy_predict_score(ep, conditions),
                    "Start": w + s + 1,
                    "End": w + e,
                    "Length": len(ep),
                    "Avg_hydrophobicity": avg_hydro(ep)
                })

    df = pd.DataFrame(records).sort_values("Score", ascending=False).reset_index(drop=True)

    if df.empty:
        st.error("No epitopes generated.")
        st.stop()

    top = df.iloc[0]

    # =============================
    # Top epitope summary
    # =============================

    st.subheader("Top-ranked Epitope Candidate")

    c1, c2, c3 = st.columns(3)
    c1.metric("Epitope", top["Epitope"])
    c2.metric("Score", f"{top['Score']:.3f}")
    c3.metric("AA position", f"{top['Start']} – {top['End']}")

    st.markdown(f"**Average hydrophobicity:** {top['Avg_hydrophobicity']:.2f}")

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

        if r.status_code != 200:
            st.warning("Failed to load PDB structure.")
            st.stop()

        pdb_text = r.text.replace("`", "")

        start = int(top["Start"])
        end = int(top["End"])

        html_code = f"""
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <div id="viewer" style="width: 720px; height: 520px;"></div>
        <script>
          let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
          viewer.addModel(`{pdb_text}`, "pdb");

          viewer.setStyle({{}}, {{
            cartoon: {{color: "lightgray", thickness: 0.4}}
          }});

          viewer.setStyle({{
            resi: [...Array({end}-{start}+1).keys()].map(i => i + {start})
          }}, {{
            cartoon: {{color: "red"}},
            stick: {{color: "red", radius: 0.45}}
          }});

          viewer.zoomTo();
          viewer.render();
        </script>
        """

        html(html_code, height=560)
