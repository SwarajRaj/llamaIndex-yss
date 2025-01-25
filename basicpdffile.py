import streamlit as st
import pdfplumber

# Function to load data from a PDF using pdfplumber
def load_pdf_data(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Streamlit file upload component
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Check if a file is uploaded
if uploaded_file is not None:
    documents = load_pdf_data(uploaded_file)
    st.write(documents)  # Display the extracted text from the PDF



