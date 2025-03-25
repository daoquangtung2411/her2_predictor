import streamlit as st
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor
from helper.fp_gen import smiles_to_ecfp, smiles_to_maccs
import datetime
from io import StringIO
import base64

xgb_model = pickle.load(open('pretrained_model/xgb_model.pkl', 'rb'))

features_col = np.load('helper/features_col.npy', allow_pickle=True)

col1, col2 = st.columns([1, 3])

with col1:
    st.image("3pp0_img.jpeg", width=140)

with col2:
    st.markdown(
    """
    <div style="text-align: left; line-height: 0.5;">
        <h1 style="margin: 0; font-size: 36px">HER2 INHIBITION PREDICTOR</h1>
        <h6>Adapt from paper: "Harnessing Machine Learning and Advanced Molecular Simulations for Discovering Novel HER2 Inhibitors"</h3>
    </div>
    """,
    unsafe_allow_html=True
	)


st.markdown("""
    🚀 **Predict HER2 pIC50 values for chemical compounds**  
    - Input a **single SMILES** string or **upload a CSV file**  
    - Download a CSV template for easy formatting  
    - Get results instantly and download predictions  
""")

# User input

smiles_placeholder = "e.g., CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl, CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4"

smiles_input = st.text_area("Enter SMILES (separate mulitple SMILES with commas)", "", placeholder=smiles_placeholder)

file_upload = st.file_uploader("Or upload a CSV file with a 'SMILES' column", type=['csv'])

example_csv = "example.csv"
with open(example_csv, 'rb') as example_file:
	b64 = base64.b64encode(example_file.read()).decode()
	href = f'<a href="data:file/csv;base64,{b64}" download="example.csv">📥 Download example CSV file</a>'

st.markdown(href, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .center-button {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stButton>button {
        background-color: purple !important;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 18px;
        transition: background-color 0.1s ease-in-out;
    }
    .stButton>button:hover {
        background-color: orange !important; /* Darker teal on hover */
        color: black
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="container">', unsafe_allow_html=True)
submit_button = st.button("Submit")
st.markdown('</div>', unsafe_allow_html=True)

result_df = None	

def predict_her2(smiles_list):

	pred_ecfp = [smiles_to_ecfp(smiles, radius=2, n_bits=2048) for smiles in smiles_list]
	pred_ecfp_df = pd.DataFrame(pred_ecfp, columns=[f'desc_ecfp_{i}' for i in range(2048)])
	
	pred_maccs = [smiles_to_maccs(smiles) for smiles in smiles_list]
	pred_maccs_df = pd.DataFrame(pred_maccs, columns=[f'desc_maccs_{i}' for i in range(167)])
	
	pred_emm_df = pd.concat([pred_ecfp_df.astype(int), pred_maccs_df], axis=1)

	X_predict = pred_emm_df[features_col]
	
	pred_val = xgb_model.predict(X_predict)

	result_df = pd.DataFrame({
		"SMILES": smiles_list, 
		"Predicted pIC50": np.round(pred_val,2), 
		"Predicted IC50 (nM)": np.round(10**(-pred_val)*10**9,2)
		})

	return result_df

if submit_button:

	result_df = None

	if smiles_input:
		smiles_list = [s.strip() for s in smiles_input.split(",") if s.strip()]
		result_df = predict_her2(smiles_list)

	elif file_upload:
		df = pd.read_csv(file_upload)
		if "SMILES" not in df.columns:
			st.error("⚠️ CSV file must contain a 'SMILES' column.")
		else:
			smiles_list = df["SMILES"].tolist()
			result_df = predict_her2(smiles_list)

	else:
		st.warning("ℹ️ Please enter SMILES or upload a file.")

if result_df is not None:

	st.subheader("📊 Prediction Results")

	result_df["Predicted pIC50"] = result_df["Predicted pIC50"].map(lambda x: f"{x:.2f}")
	result_df["Predicted IC50 (nM)"] = result_df["Predicted IC50 (nM)"].map(lambda x: f"{x:.2f}")

	st.markdown(
	    """
	    <style>
	    /* Reduce table padding for better fit */
	    .stTable tbody tr td {
	        padding: 6px 10px !important;
	    }
	    /* Make SMILES column flexible but not truncated */
	    .stTable tbody tr td:nth-child(2) {
	        max-width: 400px;
	        word-wrap: normal;
	        white-space: pre-wrap;
	    }
	    /* Bold the table headers */
    	.stTable thead th {
        font-weight: bold !important;
        text-align: center !important;
    	}
	    /* Center-align the pIC50 and IC50 values */
	    .stTable tbody tr td:nth-child(3), .stTable tbody tr td:nth-child(4) {
	        text-align: center !important;
	    }
	    </style>
	    """,
	    unsafe_allow_html=True
	)

	# Show DataFrame with better column fitting
	st.table(result_df.style.hide(axis="index"))
	csv = result_df.to_csv(index=False)
	b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
	# Create the download link with an icon
	href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv" style="text-decoration: none;">' \
	       f'📥 <strong>Download Prediction Results</strong></a>'

	st.markdown(href, unsafe_allow_html=True)




