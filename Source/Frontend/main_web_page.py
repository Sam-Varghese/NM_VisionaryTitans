import sqlite3
import streamlit as st
from jls_extract_var import option_menu
import pandas as pd
from annotated_text import annotated_text
from PIL import Image
#avni bhardwaj
#change in tab icon and title:
img = Image.open('logo_title.jpg')        
st.set_page_config(page_title="Cyber Security App", page_icon=img)    

#Removed the footer:
hide_menu_style=""" 
    <style>
    footer {visibility:hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)


def cctv_footages():
    st.title("Video Player")
    video_url = r"https://youtu.be/ZSt9tm3RoUU"

    if video_url:
        st.video(video_url)
        
def incidents_database():
    st.title("Basic Table Example")
    
    # Create a sample dataframe
    data = {
        'Name': ['John', 'Emily', 'Michael'],
        'Age': [25, 30, 28],
        'City': ['Theft', 'Stabbed', 'Robbed']
    }
    df = pd.DataFrame(data)
    
    # Display the dataframe as a table
    st.dataframe(df)        
    
def programmatic_actions():
    st.title("These are the programmatic actions")  
    
def manual_actions():
    st.title("These are the manual actions") 

st.title("Welcome to the App! ")

selected = option_menu(
            menu_title = None,
            options = ['CCTV Footages','Incidents Database', 'Programmatic Actions','Manual Actions'],
            icons =['camera-fill','database-check','chevron-double-right','clock-fill'], 
            default_index = 0,
            orientation= 'horizontal'
        )

if selected == "CCTV Footages":
    cctv_footages()
        
if selected == "Incidents Database":
    incidents_database()
        
if selected == "Programmatic Actions":
    programmatic_actions()
        
if selected == "Manual Actions":  
    manual_actions()    
    