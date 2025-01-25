# import streamlit as st

# st.page_link("Bot", label="Home", icon="🏠")
# st.page_link("youtubelink3.py", label="Page 1", icon="1️⃣")
# st.page_link("pdffile.py ", label="Page 2", icon="2️⃣", disabled=True)
# st.page_link("http://www.google.com", label="Google", icon="🌎")

# app_pages/page_link_demo.py
import streamlit as st

# app.py
import json
import streamlit as st
import requests
from streamlit_lottie import st_lottie

# Page Navigation
#Page Definitions for the Navigation Demo App
# pages = [
#     st.Page("pages/Bot.py", title="Home", icon="🏠"),
#     st.Page("pages/youtubelink3.py", title="st.navigation", icon="🧭"),
#     st.Page("pages/pdffile.py", title="st.page_link", icon="🔗"),
#     st.Page("http://www.google.com", title="st.switch_page", icon="🌎")
# ]

# # Adding pages to the sidebar navigation using st.navigation
# pg = st.navigation(pages, position="sidebar", expanded=True)
# # Running the app
# pg.run()

# def page_link():
st.title("BotPYT AI ")
st.page_link("Botyt.py", label="Gen AI Utilities : ", icon="👋")
st.page_link("pages/Bot.py", label="Home", icon="🏠")
st.page_link("pages/youtubelink3.py", label="Youtube Assistant", icon="1️⃣")
st.page_link("pages/pdffile.py ", label="PDF Assistant", icon="2️⃣")
st.page_link("http://www.google.com", label="Google", icon="🌎")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://lottie.host/736d8d5f-dcb7-4e9e-b35d-d8480bcdbec4/xK62Pg0NCn.json"

lottie_hello = load_lottieurl(lottie_url_hello)

# st.components.v1.html(lottie, width=310, height=310)

st_lottie(lottie_hello, key="hello")
# if __name__ == "__page__":
#      page_link()

