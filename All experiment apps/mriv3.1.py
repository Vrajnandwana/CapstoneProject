# biobart_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from unstructured.partition.pdf import partition_pdf
import tempfile
import speech_recognition as sr
import pyttsx3

# ---- Load environment variables ----
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYEMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYEMENT_NAME")

# ---- Session State Initialization ----
# This is crucial for remembering values between button clicks
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""
if "raw_summary" not in st.session_state:
    st.session_state["raw_summary"] = ""
if "enhanced_summary" not in st.session_state:
    st.session_state["enhanced_summary"] = ""
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None


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
    prompt = f"You are a radiologist. Validate and enhance this MRI impression draft for clinical correctness and clarity: {raw_summary}"
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYEMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content

# ---- Helper: extract findings from PDF using Unstructured ----
def extract_findings_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    elements = partition_pdf(filename=tmp_path)
    os.remove(tmp_path)
    full_text = "\n".join([el.text for el in elements if el.text])
    impression_keywords = ["Impression", "IMPRESSION"]
    split_index = len(full_text)
    for kw in impression_keywords:
        idx = full_text.find(kw)
        if idx != -1:
            split_index = min(split_index, idx)
    return full_text[:split_index].strip()

# ---- Helper: speech-to-text ----
def speech_to_text():
    recognizer = sr.Recognizer()
    # ✅ Using the default microphone as requested
    with sr.Microphone() as source:
        st.info("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.info("🎙️ Listening... Please speak now.")
        try:
            audio = recognizer.listen(source, timeout=10)
            st.info("Recognizing...")
            text = recognizer.recognize_google(audio)
            st.success("✅ Speech recognized!")
            return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected within the time limit.")
        except sr.UnknownValueError:
            st.error("❌ Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"❌ API Error: {e}")
    return None

# ---- Helper: text-to-speech ----
def text_to_audio_bytes(text: str):
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        temp_filename = fp.name
    engine.save_to_file(text, temp_filename)
    engine.runAndWait()
    with open(temp_filename, "rb") as f:
        audio_bytes = f.read()
    os.remove(temp_filename)
    return audio_bytes

# ---- Streamlit UI ----
st.title("🧠 MRI Impression Assistant")

input_option = st.radio("Select input type:", ["Text Input", "PDF Extraction", "Speech Input"])

# This local variable will hold the input for this script run
text_input_for_this_run = ""

if input_option == "Text Input":
    text_input_for_this_run = st.text_area("Paste MRI findings here:", value=st.session_state["text_input"], height=300)
    st.session_state["text_input"] = text_input_for_this_run

elif input_option == "PDF Extraction":
    pdf_file = st.file_uploader("Upload MRI PDF", type=["pdf"])
    if pdf_file:
        st.session_state["text_input"] = extract_findings_from_pdf(pdf_file)
    text_input_for_this_run = st.text_area("Edit/Add Findings:", value=st.session_state["text_input"], height=300)

elif input_option == "Speech Input":
    if st.button("🎤 Start Recording"):
        recognized_text = speech_to_text()
        if recognized_text:
            st.session_state["text_input"] = recognized_text
    text_input_for_this_run = st.text_area("Recognized/Edit Findings:", value=st.session_state["text_input"], height=300)

# ---- Generate and Enhance ----
if st.button("Generate & Enhance Impression"):
    if text_input_for_this_run.strip():
        with st.spinner("Generating raw impression..."):
            st.session_state["raw_summary"] = generate_summary(text_input_for_this_run)
        
        if AZURE_OPENAI_API_KEY:
            with st.spinner("Enhancing with GPT..."):
                st.session_state["enhanced_summary"] = enhance_with_gpt(st.session_state["raw_summary"])
        else:
            st.session_state["enhanced_summary"] = "Azure not configured."
        st.session_state["audio_bytes"] = None # Reset audio on new generation
    else:
        st.warning("Please provide findings before generating.")

# ---- Display Outputs ----
# This section is now separate, so it always shows results stored in the session state
if st.session_state["raw_summary"]:
    st.subheader("🔹 Raw Impression")
    st.text_area("", value=st.session_state["raw_summary"], height=100, key="raw_output")

if st.session_state["enhanced_summary"]:
    st.subheader("🤖 Enhanced Impression")
    st.text_area("", value=st.session_state["enhanced_summary"], height=300, key="enhanced_output")

    if st.button("🔊 Speak Enhanced Impression"):
        with st.spinner("Generating audio..."):
            st.session_state["audio_bytes"] = text_to_audio_bytes(st.session_state["enhanced_summary"])
    
    # If audio has been generated, show the player
    if st.session_state["audio_bytes"]:
        st.audio(st.session_state["audio_bytes"], format="audio/wav")