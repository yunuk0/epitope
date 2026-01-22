# VaxOptiML Streamlit Dashboard
# An integrated pipeline designed to enhance Cancer epitope prediction and prioritization

import streamlit as st
import numpy as np
import pandas as pd
import re
from typing import List, Dict

st.set_page_config(page_title="VaxOptiML", layout="wide")

# =============================
# Utility functions
# =============================

def clean_sequence(seq: str) -> str:
    seq = seq.strip()
    seq = re.sub(r">.*\n", "", seq)  # remove FASTA headers
    seq = re.sub(r"\s+", "", seq)     # remove spaces/newlines
    return seq.upper()


def sliding_window(seq: str, k: int) -> List[str]:
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


# Dummy probability model (placeholder for your trained model)
def epitope_probability(epitope: str, mhc_class: str) -> float:
    # Replace this with your real model inference
    base = sum([ord(a) for a in epitope]) % 100 / 100
    if mhc_class == "MHC-I":
        return min(1.0, base + 0.15)
    else:
        return min(1.0, base + 0.05)


# Amino acid property maps
hydrophobicity = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}

charge = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0, "Q": 0, "E": -1,
    "G": 0, "H": 0.5, "I": 0, "L": 0, "K": 1, "M": 0, "F": 0,
    "P": 0, "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0
}


# =============================
# Sidebar
# =============================

st.sidebar.title("VaxOptiML")
st.sidebar.markdown("**Epitope Screening Dashboard**")

mhc_class = st.sidebar.radio("Select MHC class", ["MHC-I", "MHC-II"])

color_scheme = st.sidebar.selectbox(
    "Choose a coloring scheme",
    ["None", "Hydrophobicity", "Charge"]
)

style_mode = st.sidebar.selectbox(
    "Residue Style",
    ["Blank", "Circle", "Circle (const)"]
)


# =============================
# Main UI
# =============================

st.title("VaxOptiML")
st.caption("An integrated pipeline designed to enhance cancer epitope prediction and prioritization")

st.markdown("""
**Instructions**  
1. Enter a protein sequence (FASTA or raw).  
2. Spaces and headers will be removed automatically.  
3. If sequence length > 250 aa, use the standalone package.  
4. Do not close the browser while analysis is running.
""")

sequence_input = st.text_area("Enter Protein Sequence", height=200)

run = st.button("Run Epitope Screening")

# =============================
# Analysis
# =============================

if run and sequence_input:
    seq = clean_sequence(sequence_input)

    if len(seq) > 250:
        st.warning("Sequence longer than 250 aa. Please use the standalone package.")
    else:
        st.success(f"Sequence length: {len(seq)} aa")

        epitope_length = 9 if mhc_class == "MHC-I" else 15
        epitopes = sliding_window(seq, epitope_length)

        results = []
        for i, e in enumerate(epitopes):
            prob = epitope_probability(e, mhc_class)
            results.append({
                "Epitope": e,
                "Start": i + 1,
                "End": i + epitope_length,
                "MHC": mhc_class,
                "Probability": prob
            })

        df = pd.DataFrame(results).sort_values("Probability", ascending=False)
        st.subheader("Top Predicted Epitopes")
        st.dataframe(df.head(20), use_container_width=True)

        # =============================
        # Antigen highlight
        # =============================
        st.subheader("Antigen Sequence with Epitope Highlight")

        top_epitopes = df.head(10)
        highlight = {i: 0 for i in range(len(seq))}

        for _, row in top_epitopes.iterrows():
            for pos in range(row.Start - 1, row.End):
                highlight[pos] = max(highlight[pos], row.Probability)

        def color_residue(i, aa):
            base_color = "255,255,255"

            if color_scheme == "Hydrophobicity":
                val = hydrophobicity.get(aa, 0)
                norm = (val + 4.5) / 9
                r = int(255 * norm)
                b = int(255 * (1 - norm))
                base_color = f"{r},100,{b}"
            elif color_scheme == "Charge":
                val = charge.get(aa, 0)
                norm = (val + 1) / 2
                r = int(255 * norm)
                b = int(255 * (1 - norm))
                base_color = f"{r},100,{b}"

            alpha = highlight[i]
            shape = "border-radius:50%;" if "Circle" in style_mode else ""
            size = "padding:4px;" if style_mode == "Circle" else "padding:2px;"

            return f"<span style='background-color: rgba({base_color},{alpha}); {shape} {size} font-family:monospace'>{aa}</span>"

        html_seq = "".join([color_residue(i, aa) for i, aa in enumerate(seq)])
        st.markdown(html_seq, unsafe_allow_html=True)

        # =============================
        # Probability distribution (Streamlit native)
        # =============================
        st.subheader("Epitope Probability Distribution")
        st.bar_chart(df["Probability"])
