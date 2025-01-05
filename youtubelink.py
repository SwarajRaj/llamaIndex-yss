import re
import os
import traceback
import streamlit as st
from tqdm import tqdm

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

GROQ_API_KEY = "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"

PROMPT_TEMPLATE = (
    "You are an expert in answering questions related to current affairs."
    "Context information is below which is extracted from news websites.\n"
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
st.title("Ask LlamaIndex")
user_query1 = st.text_input("Enter a link.......")
user_query = st.text_input("What would you like to ask?")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not user_query.strip():
        st.error(f"Please provide the search query.")
    elif not user_query1.strip():  # Check if user_query1 is empty
        st.error(f"Please provide a YouTube link.") 
    else:
        try:
            # Now we can call the function to get the valid URL
            #valid_url = validate_and_format_youtube_url(user_input_url=user_query1)
            
            # Initialize the YoutubeTranscriptReader here
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(ytlinks = [user_query1])

            # text splitter
            text_parser = TokenTextSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            chunks = text_parser.split_text(text=documents[0].text)

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
            st.write(response.response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(traceback.format_exc())