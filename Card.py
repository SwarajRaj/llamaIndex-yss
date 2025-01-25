import streamlit as stm 
from streamlit_card import card 
from PIL import Image
from streamlit_extras.let_it_rain import rain 
 

logo=Image.open(r'C:\Users\seshu\Desktop\llama_index_Pro\google.jpg')
# img=Image.open(r'C:\Users\HP\Pictures\Harshimg.jpg')
# st.image(img,width=200,caption="Harsh image")

stm.set_page_config(page_title="This is a Simple Streamlit WebApp") 
stm.title("This is the Home Page Geeks.") 
stm.text("Geeks Home Page") 


# Card 


card( 
	title="Hello Geeks!", 
	text="Click this card to redirect to GeeksforGeeks", 
	image="https://yt3.googleusercontent.com/g_bEA4DiQjWzCdRluwELXUOZ4zWelOaz_sFb61X6S2swcVTuGevoD1v-MDFZ0WS44IZ4zjNPEg=s160-c-k-c0x00ffffff-no-rj", 
	url="https://www.geeksforgeeks.org/", 
) 

  
rain( 
    emoji="🌎", 
    font_size=40,  # the size of emoji 
    falling_speed=3,  # speed of raining 
    animation_length="infinite",  # for how much time the animation will happen 
) 