import streamlit as st
import requests

st.set_page_config(page_title="NovaDocs AI", layout="wide")

st.title("🚀 NovaDocs AI")

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "txt", "md", "csv"]
)

question = st.text_input("Ask a question")

if st.button("Ask"):
    st.write("Processing...")