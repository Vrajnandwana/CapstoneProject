# biobart_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from unstructured.partition.pdf import partition_pdf
import tempfile
import re

# ---- Load environment variables ----
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# ---- Session State Initialization ----
if "raw_impression" not in st.session_state:
    st.session_state.raw_impression = ""
if "enhanced_impression" not in st.session_state:
    st.session_state.enhanced_impression = ""

# ---- Model config ----
# Using your Hugging Face model ID for deployment
#MODEL_DIR = "Vrajk/mri-impressions" 
MODEL_DIR = "./biobart-mri"
DEVICE = "cpu"

@st.cache_resource(show_spinner="Loading Mri Impressions Generation model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        device_map=None,
        dtype=torch.float32
    )
    model.to(DEVICE)
    return tokenizer, model

tokenizer, model = load_model()

# ---- Helper functions ----
def generate_impression(text, min_len, max_len, beams):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        min_length=min_len,
        num_beams=beams,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def enhance_with_gpt(raw_impression: str):
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    prompt = f"Refine this MRI impression draft for clarity and clinical correctness, returning only the final text: {raw_impression}"
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

def extract_findings_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    elements = partition_pdf(filename=tmp_path)
    os.remove(tmp_path)
    full_text = "\n".join([el.text for el in elements if el.text])
    impression_keywords = ["Impression", "IMPRESSION"]
    findings_start_patterns = ["SEQUENCES:", "HISTORY:", "FINDINGS:"]
    start_idx = 0
    for pat in findings_start_patterns:
        idx = full_text.find(pat)
        if idx != -1:
            start_idx = idx + len(pat)
            break
    end_idx = len(full_text)
    for kw in impression_keywords:
        idx = full_text.find(kw)
        if idx != -1:
            end_idx = min(end_idx, idx)
    findings_text = full_text[start_idx:end_idx].strip()
    return re.sub(r"(\n\s*)+\n+", "\n", findings_text).strip()

# ---- Streamlit UI ----
st.set_page_config(page_title="MRI Impression Assistant", layout="wide")

st.markdown("<h1 style='text-align:center; color:#4B0082;'>🧠 MRI Impression Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---- CHANGE: Sidebar now only has advanced settings ----
with st.sidebar:
    with st.expander("⚙️ Advanced Model Settings"):
        beam_size = st.slider("Beam Size", min_value=2, max_value=10, value=5)
        min_len = st.number_input("Min Impression Length", min_value=20, max_value=100, value=60)
        max_len = st.number_input("Max Impression Length", min_value=80, max_value=300, value=160)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["💡 Input", "🔹 Raw Impression", "🤖 Enhanced Impression"])

with tab1:
    st.header("Input MRI Findings")
    st.info("Follow the steps below to generate a report.", icon="ℹ️")

    # ---- CHANGE: Input options are now in the main tab ----
    st.subheader("Step 1: Choose Your Input Method")
    input_option = st.radio(
        "Select Input Type:", 
        ["Text Input", "PDF Upload"], 
        horizontal=True, 
        label_visibility="collapsed"
    )

    st.subheader("Step 2: Provide the Findings")
    text_input = ""
    if input_option == "Text Input":
        text_input = st.text_area("Paste MRI Findings Here:", height=250, placeholder="Enter MRI findings...")
    elif input_option == "PDF Upload":
        pdf_file = st.file_uploader("Upload an MRI Report PDF", type=["pdf"])
        if pdf_file:
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_findings_from_pdf(pdf_file)
                st.success("✅ Findings extracted from PDF")
                text_input = st.text_area("Edit or Add to the Extracted Findings:", value=extracted_text, height=250)
    
    st.subheader("Step 3: Generate the Report")
    if st.button("Generate Impressions", key="generate_btn"):
        if text_input and text_input.strip():
            with st.spinner("Generating raw impression..."):
                st.session_state.raw_impression = generate_impression(text_input, min_len=min_len, max_len=max_len, beams=beam_size)
            
            if AZURE_OPENAI_API_KEY:
                with st.spinner("Enhancing impression with GPT..."):
                    st.session_state.enhanced_impression = enhance_with_gpt(st.session_state.raw_impression)
            
            st.toast("Impressions generated successfully!", icon="🎉")
        else:
            st.warning("Please provide MRI findings before generating.")

with tab2:
    st.header("Raw Impression (BioBART)")
    if st.session_state.raw_impression:
        st.markdown(f"<div style='background-color:#E6E6FA; padding:15px; border-radius:10px;'>{st.session_state.raw_impression}</div>", unsafe_allow_html=True)
    else:
        st.info("The raw impression generated by the BioBART model will appear here.")

with tab3:
    st.header("Enhanced Impression (GPT)")
    if st.session_state.enhanced_impression:
        st.markdown(f"<div style='background-color:#D8F6CE; padding:15px; border-radius:10px;'>{st.session_state.enhanced_impression}</div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Enhanced Impression",
            data=st.session_state.enhanced_impression,
            file_name="enhanced_mri_impression.txt",
            mime="text/plain"
        )
    elif not AZURE_OPENAI_API_KEY:
         st.warning("Azure OpenAI not configured. Add secrets to enable enhancement.")
    else:
        st.info("The final, enhanced impression from GPT will appear here after generation.")

# (Footer code can remain as is)
with st.expander("ℹ️ About / Instructions"):
    st.markdown("""
- Paste MRI findings manually or upload a PDF containing MRI reports.
- BioBART generates a raw impression summary.
- GPT validates and enhances the impression into a professional report.
- Use the tabs to navigate between input, raw summary, and enhanced output.
- Copy or download the enhanced impression for documentation.
""")