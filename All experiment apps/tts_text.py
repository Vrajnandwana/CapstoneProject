# tts_test.py
import streamlit as st
import pyttsx3
import tempfile
import os

def text_to_audio_bytes(text: str):
    """
    Converts text to speech and returns the audio as bytes.
    This function includes print statements for debugging.
    """
    try:
        print("1. Initializing pyttsx3 engine...")
        engine = pyttsx3.init()

        # Use a temporary file to save the speech as a WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            temp_filename = fp.name

        print(f"2. Saving speech to temporary file: {temp_filename}")
        engine.save_to_file(text, temp_filename)

        print("3. Running engine.runAndWait() to process the file...")
        engine.runAndWait()

        print("4. Reading audio bytes from the file...")
        with open(temp_filename, "rb") as f:
            audio_bytes = f.read()

        os.remove(temp_filename) # Clean up the temp file

        print("5. Success! Returning audio bytes.")
        return audio_bytes

    except Exception as e:
        print(f"An error occurred in TTS generation: {e}")
        return None

st.title("Text-to-Speech Test")

text_to_speak = st.text_input("Enter text to speak:", "Hello, can you hear me now?")

if st.button("Generate and Play Audio"):
    if text_to_speak:
        st.info("Generating audio... check your terminal for progress.")
        audio_data = text_to_audio_bytes(text_to_speak)

        if audio_data:
            st.success("Audio generated! Playing now.")
            st.audio(audio_data, format="audio/wav")
        else:
            st.error("Failed to generate audio. Check the terminal for errors.")
    else:
        st.warning("Please enter some text.")