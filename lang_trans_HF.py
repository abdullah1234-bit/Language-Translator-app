import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Streamlit UI setup
st.set_page_config(page_title="🌐 Hugging Face Translator", layout="centered")
st.title("🌍 Language Translator using Hugging Face")

# Input area
input_text = st.text_area("Enter text to translate (multilingual to English):")

# Button
translate_btn = st.button("Translate")

# Model to use
MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"

# Load Hugging Face client (cached for performance)
@st.cache_resource
def load_client():
    return InferenceClient(token=HF_TOKEN)

client = load_client()

# Translation function
def translate_text(text: str) -> str:
    try:
        result = client.translation(text, model=MODEL_NAME)
        return result
    except Exception as e:
        return f"❌ Error: {e}"

# Trigger translation on button click
if translate_btn and input_text:
    with st.spinner("Translating..."):
        result = translate_text(input_text)
        st.success("Translation complete!")
        st.markdown("### 📝 Translated Text:")
        st.write(result)
