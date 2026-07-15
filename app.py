import streamlit as st
from rag_pipeline import rag_answer

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📚 RAG Chatbot")
st.write("Ask questions based on your uploaded documents.")

# Chat input
query = st.text_input("Your question")

if query:
    with st.spinner("Thinking..."):
        answer = rag_answer(query)
    st.write("### Answer")
    st.write(answer)
