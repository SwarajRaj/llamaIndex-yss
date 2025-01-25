import re
import os
import traceback
import streamlit as st
from tqdm import tqdm
import pdfplumber
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TextNode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import get_response_synthesizer
from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from langchain.chat_models import init_chat_model


import time
import requests

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

GROQ_API_KEY = "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"

PROMPT_TEMPLATE = (
    "You are an expert in answering questions related to current affairs."
    "Context information is below content provide recommended follow-up questions more than or equal to two at last with italic bold heading."
    "Context information is below content is pdf transcript.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the question asked by the user."
    "Include useful URLs in the response."
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# load the embedding model from hugging face
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create LLM object
llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

# Define a simple Streamlit app
st.title("BotYT AI")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # You can process the file here, for example:
    # st.write(uploaded_file.name)  # Display the name of the uploaded file

    # Success message
    st.success(f"Successfully uploaded: {uploaded_file.name}")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://lottie.host/e662f0ae-d902-4bb1-8620-4250557d3a41/9K6fWSOebi.json"

lottie_hello = load_lottieurl(lottie_url_hello)
st_lottie(lottie_hello, key="hello", width=700, height=200)

# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


# user_query = st.text_input("What would you like to ask?")

# # If the 'Submit' button is clicked
# if st.button("Submit"):
#     if not user_query.strip():
#         st.error(f"Please provide the search query.")
#     elif not user_query1.strip():  # Check if user_query1 is empty
#         st.error(f"Please provide a YouTube link.") 
#     else:
#         try:
            # Now we can call the function to get the valid URL
            #valid_url = validate_and_format_youtube_url(user_input_url=user_query1)
            
            # Initialize the YoutubeTranscriptReader here
def load_pdf_data(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text
def query_response(user_query):
        #loader = YoutubeTranscriptReader()
        documents = load_pdf_data(uploaded_file)

                    # text splitter
        text_parser = TokenTextSplitter(
                chunk_size=512,
                chunk_overlap=50
              )
        chunks = text_parser.split_text(text=documents)

                    # convert chunks into llama nodes
        nodes = [TextNode(text=chunk_text) for chunk_text in chunks]

                    # Create embeddings for the chunks
        for node in tqdm(nodes):
            node.embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )

                    # index the data
        index = VectorStoreIndex(
                nodes=nodes, embed_model=embed_model
            )

                    # Create a retriever object
        retriever = index.as_retriever(similarity_top_k=3)

                    # Create prompt
        qa_template = PromptTemplate(PROMPT_TEMPLATE)

                    # configure response synthesizer
        response_synthesizer = get_response_synthesizer(llm, text_qa_template = qa_template)


                    # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0)]
        )
                    
        response = query_engine.query(user_query)
        return response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # response = f"Echo: {query_response(prompt)}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = query_response(prompt)
    #     message_placeholder = st.empty()
    #     full_response = ""
    #     for chunk in response:
    #         full_response += chunk
    #         message_placeholder.markdown(full_response + "▌")
    # message_placeholder.markdown(full_response)
        # print(response)

# Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

        # except Exception as e:
        #     st.error(f"An error occurred: {e}")
        #     st.error(traceback.format_exc())


import streamlit as st

# Function to save content to a file
def save_to_file(content):
    with open('feedbacks.txt', 'a') as f:
        f.write(content + '\n')
        


# Like button
if st.button(f"👍 Like"):
    # Do something for like
    
    save_to_file(" Like")

# Dislike button
if st.button(f"👎 Dislike"):
    # Do something for dislike
    
    save_to_file(" Dislike")

# Collect feedback from the user
feedback_text = st.text_input("Submit Feedback:")

# Collect rating from the user
# rating = st.slider("Rate your experience (1-5)", 1, 5, 3)
# rating = st.number_input("Rate your experience", min_value=1, max_value=5, value=3)
# rating = st.radio("Rate your experience", [1, 2, 3, 4, 5])
rating = st.radio("Rate your experience", ["★", "★★", "★★★", "★★★★", "★★★★★"])
# rating = st.selectbox("Rate your experience", [1, 2, 3, 4, 5])

# Submit feedback with rating
if st.button("Submit Feedback"):
    print(feedback_text)
    save_to_file(f"Feedback: {feedback_text}")
    save_to_file(f"Rating: {rating}")
    save_to_file("-----------------")
else:
    # Expecting feedback on every answer
    st.stop()
