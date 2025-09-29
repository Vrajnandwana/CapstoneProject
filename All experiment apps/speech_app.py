# speech_app.py
import streamlit as st
import speech_recognition as sr
import pyttsx3
import tempfile
import os

# ---- Text-to-Speech Function (Proven to work) ----
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

# ---- Speech-to-Text Function ----
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Adjusting for ambient noise... please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        st.info("Listening... please speak into your microphone.")
        try:
            audio = recognizer.listen(source, timeout=5) # Listen for 5 seconds
            st.info("Recognizing speech...")
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected within the time limit.")
            return None
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google API; {e}")
            return None

# ---- Streamlit UI ----
st.title("🎤 Speech-to-Text and Text-to-Speech 🗣️")

if st.button("Start Listening & Get Speech"):
    recognized_text = recognize_speech_from_mic()

    if recognized_text:
        st.success(f"**Recognized Text:** {recognized_text}")

        st.info("Converting recognized text to speech...")
        audio_data = text_to_audio_bytes(recognized_text)

        if audio_data:
            st.audio(audio_data, format="audio/wav")
        else:
            st.error("Failed to generate audio for the recognized text.")