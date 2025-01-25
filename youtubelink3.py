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
    "Context information is below content is youtube transcript.\n"
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
st.title("BotPYT AI")
user_query1 = st.text_input("Enter a link.......")
# user_query = st.text_input("What would you like to ask?")
if user_query1 :
    # You can process the file here, for example:
    # st.write(uploaded_file.name)  # Display the name of the uploaded file

    # Success message
    st.success(f"Successfully uploaded link......")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://lottie.host/9f7d9323-5229-435f-ab85-28ac682fbc4f/Tdcd3AmUQ6.json"

lottie_hello = load_lottieurl(lottie_url_hello)
st_lottie(lottie_hello, key="hello", width=700, height=200)


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
def query_response(user_query):
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

# def query_refiner(conversation, query):
#     response = openai.Completion.create(
#     model="text-davinci-003",
#     prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#     temperature=0.7,
#     max_tokens=256,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return response['choices'][0]['text']

# def find_match(input):
#     input_em = model.encode(input).tolist()
#     result = index.query(input_em, top_k=2, includeMetadata=True)
#     return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

# def get_conversation_string():
#     conversation_string = ""
#     for i in range(len(st.session_state['responses'])-1):        
#         conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
#         conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
#     return conversation_string

# import streamlit as st
# if st.button(f"👍 Like"):
#     #do something
#     print("hi")
# if st.button(f"👎 Dislike"):
#     #do something
#     print("bye")

# feedback_text = st.text_input("Submit Feedback:")
# if st.button("Submit Feedback"):
#     print(feedback_text)
# else:
#     # expecting feedback on every answer
#     st.stop()

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
rating = st.number_input("Rate your experience", min_value=1, max_value=5, value=3)
# rating = st.radio("Rate your experience", [1, 2, 3, 4, 5])
#rating = st.radio("Rate your experience", ["★", "★★", "★★★", "★★★★", "★★★★★"])
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
