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
import threading
import queue

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
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ---- Azure GPT Helper ----
def enhance_with_gpt(raw_summary: str):
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    prompt = f"""
    You are a radiologist. Here is an MRI impression draft:

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

# ---- Helper: extract findings from PDF using Unstructured ----
def extract_findings_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    elements = partition_pdf(filename=tmp_path)
    full_text = "\n".join([el.text for el in elements if el.text])
    
    impression_keywords = ["Impression", "IMPRESSION"]
    split_index = len(full_text)
    for kw in impression_keywords:
        idx = full_text.find(kw)
        if idx != -1:
            split_index = min(split_index, idx)
    findings_text = full_text[:split_index].strip()
    return findings_text

# ---- Session State Initialization ----
if "speech_text" not in st.session_state:
    st.session_state["speech_text"] = ""
if "recording" not in st.session_state:
    st.session_state["recording"] = False
if "audio_queue" not in st.session_state:
    st.session_state["audio_queue"] = queue.Queue()
# --- STATE MANAGEMENT FIX: Add state for summaries ---
if "raw_summary" not in st.session_state:
    st.session_state["raw_summary"] = ""
if "enhanced_summary" not in st.session_state:
    st.session_state["enhanced_summary"] = ""
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None


# ---- Continuous speech-to-text ----
def continuous_speech_recognition(audio_queue):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        while st.session_state.get("recording", False):
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=10)
                text = recognizer.recognize_google(audio)
                audio_queue.put(text)
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except sr.RequestError as e:
                audio_queue.put(f"[ERROR] API error: {e}")
                st.session_state["recording"] = False

# ---- TTS FIX: Save to file and use st.audio ----
def text_to_audio_bytes(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        engine.save_to_file(text, fp.name)
        engine.runAndWait()
        fp.seek(0)
        return fp.read()

# ---- Streamlit UI ----
st.title("🧠 MRI Impression Assistant")

input_option = st.radio("Select input type:", ["Text Input", "PDF Extraction", "Speech Input"])
text_input = ""

# --- Text Input ---
if input_option == "Text Input":
    text_input = st.text_area("Paste MRI findings here:", height=300)

# --- PDF Extraction ---
elif input_option == "PDF Extraction":
    pdf_file = st.file_uploader("Upload MRI PDF", type=["pdf"])
    if pdf_file:
        extracted_text = extract_findings_from_pdf(pdf_file)
        st.info("Text extracted. You can now edit or add findings.")
        text_input = st.text_area("Edit/Add Findings:", value=extracted_text, height=300)

# --- Speech Input ---
elif input_option == "Speech Input":
    # Consume queue to update text
    while not st.session_state["audio_queue"].empty():
        st.session_state["speech_text"] += " " + st.session_state["audio_queue"].get()

    text_input = st.text_area("Recognized Text:", value=st.session_state["speech_text"], height=300)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("▶️ Start Recording"):
            if not st.session_state["recording"]:
                st.session_state["recording"] = True
                threading.Thread(target=continuous_speech_recognition, args=(st.session_state["audio_queue"],), daemon=True).start()
                st.rerun() # Force a rerun to update status
    with col2:
        if st.button("⏸ Pause Recording"):
            st.session_state["recording"] = False
            st.rerun()
    with col3:
        if st.button("🔄 Reset Recording"):
            st.session_state["recording"] = False
            st.session_state["speech_text"] = ""
            # --- SPEECH INPUT FIX: Clear queue on Reset ---
            with st.session_state["audio_queue"].mutex:
                st.session_state["audio_queue"].queue.clear()
            st.rerun()
    
    st.write(f"Recording Status: {'Active' if st.session_state.get('recording') else 'Inactive'}")

# ---- Generate & Enhance Button ----
if st.button("Generate & Enhance Impression"):
    if text_input.strip():
        with st.spinner("Generating raw impression..."):
            st.session_state["raw_summary"] = generate_summary(text_input)
        
        if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
            with st.spinner("Enhancing impression with GPT..."):
                st.session_state["enhanced_summary"] = enhance_with_gpt(st.session_state["raw_summary"])
        else:
            st.session_state["enhanced_summary"] = "Azure not configured."
        
        # Clear any previous audio
        st.session_state["audio_bytes"] = None

    else:
        st.warning("Please provide MRI findings before generating.")

# ---- UI LOGIC FIX: Always display summaries if they exist in session_state ----
if st.session_state["raw_summary"]:
    st.subheader("🔹 Raw Impression")
    st.text_area("Raw Impression", value=st.session_state["raw_summary"], height=100, key="raw_sum_area")

if st.session_state["enhanced_summary"]:
    st.subheader("🤖 Enhanced Impression")
    st.text_area("Enhanced Impression", value=st.session_state["enhanced_summary"], height=300, key="enh_sum_area")

    # TTS Speaker button
    if st.button("🔊 Speak Enhanced Impression"):
        with st.spinner("Generating audio..."):
            st.session_state["audio_bytes"] = text_to_audio_bytes(st.session_state["enhanced_summary"])
    
    # Display audio player if audio bytes exist
    if st.session_state["audio_bytes"]:
        st.audio(st.session_state["audio_bytes"], format="audio/mp3")