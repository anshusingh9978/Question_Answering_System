import streamlit as st
from transformers import pipeline
import wikipedia
import fitz  # PyMuPDF
from gtts import gTTS
from io import BytesIO
import base64
import os
import speech_recognition as sr
from pydub import AudioSegment
from audiorecorder import audiorecorder

# ✅ FFmpeg setup
ffmpeg_bin = r"C:\Users\sansh\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_bin
AudioSegment.converter = os.path.join(ffmpeg_bin, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_bin, "ffprobe.exe")

@st.cache_resource
def load_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_model = load_model()

st.set_page_config(page_title="Ultimate QA App", layout="centered")
st.title("🧠 Question Answering App")

# Wikipedia Search
st.header("🔍 Wikipedia Search")
wiki_query = st.text_input("Search a Wikipedia topic:")
wiki_text = ""
if wiki_query:
    try:
        wiki_text = wikipedia.page(wiki_query).content
        st.success("✅ Wikipedia content loaded.")
    except Exception as e:
        st.error(f"Wikipedia Error: {e}")

# Manual Text
st.header("📝 Enter Your Own Text")
user_paragraph = st.text_area("Paste or type your custom paragraph:")

# Upload File
st.header("📄 Upload a .pdf or .txt File")
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])
file_text = ""
if uploaded_file:
    ext = uploaded_file.name.split('.')[-1]
    if ext == "txt":
        file_text = uploaded_file.read().decode("utf-8")
    elif ext == "pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                file_text += page.get_text()
    st.success("✅ File content extracted.")

# Select context
st.header("📚 Select Context Source")
option = st.radio("Choose the context you want to use:", ("Wikipedia", "Entered Text", "Uploaded File"))
context = {"Wikipedia": wiki_text, "Entered Text": user_paragraph, "Uploaded File": file_text}.get(option, "")

# ❓ Ask Question
st.subheader("❓ Ask Your Question (Type or Record)")
question = st.text_input("Type your question here:")

# 🎙️ Audio Recorder
st.markdown("Or record your voice:")
audio = audiorecorder("Click to record", "Recording...", key="qa_audio")
voice_question = ""

if len(audio) > 0:
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    wav_bytes = wav_io.getvalue()
    st.audio(wav_bytes, format="audio/wav")

    with open("voice_input.wav", "wb") as f:
        f.write(wav_bytes)

    recognizer = sr.Recognizer()
    with sr.AudioFile("voice_input.wav") as source:
        audio_data = recognizer.record(source)
        try:
            voice_question = recognizer.recognize_google(audio_data)
            st.success(f"🎤 Recognized: {voice_question}")
            question = voice_question
        except sr.UnknownValueError:
            st.warning("😕 Couldn't understand audio.")
        except sr.RequestError:
            st.error("🚫 Speech service unavailable.")

# Preview and Run Model
if context:
    st.subheader("📖 Context Preview")
    st.text_area("Context in use:", value=context, height=200)

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter or record a question.")
        else:
            with st.spinner("Finding the answer..."):
                try:
                    result = qa_model(question=question, context=context)
                    st.success(f"✅ Answer: {result['answer']}")
                    st.caption(f"Confidence: {result['score']:.2f}")

                    # 🔊 Voice Output
                    tts = gTTS(result["answer"])
                    tts.save("answer.mp3")
                    with open("answer.mp3", "rb") as f:
                        st.audio(f.read(), format="audio/mp3")
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("ℹ️ Please select or enter a context before asking a question.")

