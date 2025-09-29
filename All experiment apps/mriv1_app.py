# biobart_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import os
from openai import AzureOpenAI

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
        dtype=torch.float32   # ✅ safe for CPU
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
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ---- Azure GPT Helper ----
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
        max_tokens=300
    )
    return response.choices[0].message.content

# ---- Streamlit UI ----
st.title("🧠 BioBART MRI Impression Assistant")

text_input = st.text_area("Paste MRI findings here:", height=300)

if st.button("Generate Impression"):
    if text_input.strip():
        # Step 1: BioBART
        raw_summary = generate_summary(text_input)
        st.subheader("🔹 Raw Impression (BioBART)")
        st.text_area("Raw Impression", value=raw_summary, height=200)

        # Step 2: Azure GPT Validation & Completion
        if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
            enhanced_summary = enhance_with_gpt(raw_summary)
            st.subheader("🤖 Enhanced Impression (with GPT)")
            st.text_area("Enhanced Impression", value=enhanced_summary, height=200)
        else:
            st.warning("⚠️ Azure OpenAI is not configured in .env. Skipping enhancement.")
    else:
        st.warning("Please paste MRI findings before generating.")
