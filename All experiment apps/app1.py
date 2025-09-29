import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model + tokenizer from saved directory
MODEL_DIR = "./biobart-mri"   # your saved folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# Auto device selection (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Page title
st.set_page_config(page_title="MRI Report Summarizer", layout="wide")
st.title("MRI Report Summarizer")

# Layout: Two columns side by side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter MRI Findings")
    findings = st.text_area("MRI Findings:", height=400, key="findings_text")

    if st.button("Generate Impression"):
        if findings.strip():
            inputs = tokenizer(
                findings,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding="max_length"
            ).to(device)

            with st.spinner("Generating..."):
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=160,
                    min_length=40,
                    num_beams=4,
                    early_stopping=True
                )
                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            st.session_state["generated_output"] = output
        else:
            st.warning(" Please enter some findings text!")

with col2:
    st.subheader("Generated Impression")
    if "generated_output" in st.session_state:
        st.success(st.session_state["generated_output"])
    else:
        st.info("Impression will appear here after you click **Generate**.")
