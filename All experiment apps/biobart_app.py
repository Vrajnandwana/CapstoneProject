# biobart_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
import torch
from dotenv import load_dotenv
import os
import tempfile
from PyPDF2 import PdfReader

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
        device_map=None,   # ensures CPU
        torch_dtype=torch.float32
    )
    model.to(DEVICE)
    return tokenizer, model

tokenizer, model = load_model()

# ---- Helper: generate summary ----
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
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# ---- Streamlit UI ----
st.title("🧠 BioBART MRI Impression Generator")

# Input choice
input_option = st.radio("Select input type:", ["Text Input", "PDF Upload"])

text_input = ""
if input_option == "Text Input":
    text_input = st.text_area("Paste MRI findings here:", height=300)
elif input_option == "PDF Upload":
    pdf_file = st.file_uploader("Upload MRI PDF", type=["pdf"])
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        reader = PdfReader(tmp_path)
        text_input = ""
        for page in reader.pages:
            text_input += page.extract_text() + "\n"

if text_input.strip():
    st.markdown("### Generated Impression:")
    raw_summary = generate_summary(text_input)
    st.text_area("Impression", value=raw_summary, height=200)

    # Optional: ground truth evaluation
    reference_summary = st.text_area("Enter ground-truth impression (optional):", height=150)
    if reference_summary.strip():
        rouge = load("rouge")
        scores = rouge.compute(predictions=[raw_summary], references=[reference_summary])
        st.markdown("### 📊 ROUGE Scores:")
        for k, v in scores.items():
            if hasattr(v, "mid"):
                st.write(f"{k}: {v.mid.fmeasure:.4f}")
            else:
                st.write(f"{k}: {v:.4f}")

    # Optional: integrate Azure OpenAI for enhancing impression
    if AZURE_OPENAI_API_KEY and st.button("Enhance with GPT"):
        from openai import OpenAI
        client = OpenAI(api_key=AZURE_OPENAI_API_KEY)
        prompt = f"Improve the MRI impression below for clinical clarity:\n\n{raw_summary}"
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYEMENT_NAME,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=250
        )
        enhanced_summary = response.choices[0].message.content
        st.markdown("### 🤖 Enhanced Impression:")
        st.text_area("Enhanced Impression", value=enhanced_summary, height=200)
