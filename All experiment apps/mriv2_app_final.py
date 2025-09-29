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
AZURE_OPENAI_DEPLOYEMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYEMENT_NAME")

# ---- Model config ----
MODEL_DIR = "./biobart-mri"
DEVICE = "cpu"

@st.cache_resource(show_spinner=True)
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
def generate_summary(text, min_len=40, max_len=160, beams=4):
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

def enhance_with_gpt(raw_summary: str):
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    prompt = f"""
    You are a medical assistant. Here is an MRI impression draft:

    {raw_summary}

    Validate it for clinical correctness and complete any missing or important details
    so that it becomes a clear, professional radiology impression.
    """
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYEMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content

def extract_findings_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    elements = partition_pdf(filename=tmp_path)
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
    findings_text = re.sub(r"\n+", "\n", findings_text)
    findings_text = re.sub(r"[ \t]+", " ", findings_text)

    return findings_text

# ---- Streamlit UI ----
st.set_page_config(page_title="BioBART MRI Assistant", layout="wide")

st.markdown("<h1 style='text-align:center; color:#4B0082;'>🧠 BioBART MRI Impression Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar: Model Settings
st.sidebar.title("⚙️ Model Settings")
beam_size = st.sidebar.slider("Beam Size", min_value=2, max_value=10, value=4)
min_len = st.sidebar.number_input("Min Summary Length", min_value=20, max_value=100, value=40)
max_len = st.sidebar.number_input("Max Summary Length", min_value=80, max_value=300, value=160)

# Sidebar: Input type
st.sidebar.title("📝 Input Options")
input_option = st.sidebar.radio("Select Input Type:", ["Text Input", "PDF Upload"])

# Tabs for organized UI
tab1, tab2, tab3 = st.tabs(["💡 Input", "🔹 Raw Impression", "🤖 Enhanced Impression"])

# ---- Input Tab ----
with tab1:
    st.header("Input MRI Findings")
    st.info("Paste MRI findings manually or upload a PDF to extract automatically.", icon="ℹ️")

    text_input = ""
    if input_option == "Text Input":
        text_input = st.text_area("Paste MRI Findings Here:", height=250, placeholder="Enter MRI findings...")
    elif input_option == "PDF Upload":
        pdf_file = st.file_uploader("Upload MRI PDF", type=["pdf"])
        if pdf_file:
            extracted_text = extract_findings_from_pdf(pdf_file)
            st.success("✅ Findings extracted from PDF")
            text_input = st.text_area("Edit/Add Findings:", value=extracted_text, height=250)

    generate_btn = st.button("Generate Impressions", key="generate_btn")

# ---- Raw Impression Tab ----
raw_summary = ""
if generate_btn and text_input.strip():
    with tab2:
        st.header("Raw Impression (BioBART)")
        raw_summary = generate_summary(text_input, min_len=min_len, max_len=max_len, beams=beam_size)
        st.markdown(f"<div style='background-color:#E6E6FA; padding:15px; border-radius:10px;'>{raw_summary}</div>", unsafe_allow_html=True)
        st.button("Copy Raw Impression", key="copy_raw")
        st.success("✅ Raw impression generated successfully!")

# ---- Enhanced Impression Tab ----
if generate_btn and text_input.strip() and AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
    with tab3:
        st.header("Enhanced Impression (GPT)")
        enhanced_summary = enhance_with_gpt(raw_summary)
        st.markdown(f"<div style='background-color:#D8F6CE; padding:15px; border-radius:10px;'>{enhanced_summary}</div>", unsafe_allow_html=True)
        st.button("Copy Enhanced Impression", key="copy_enhanced")
        st.download_button(
            label="📥 Download Enhanced Impression",
            data=enhanced_summary,
            file_name="enhanced_mri_impression.txt",
            mime="text/plain"
        )
        st.success("🎉 Enhanced impression generated successfully!")

# ---- Footer / About ----
with st.expander("ℹ️ About / Instructions"):
    st.markdown("""
- Paste MRI findings manually or upload a PDF containing MRI reports.
- BioBART generates a raw impression summary.
- GPT validates and enhances the impression into a professional report.
- Use the tabs to navigate between input, raw summary, and enhanced output.
- Copy or download the enhanced impression for documentation.
""")
