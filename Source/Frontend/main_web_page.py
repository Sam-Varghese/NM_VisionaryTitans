import sqlite3
import streamlit as st
import hydralit_components as hc
import datetime
#------------------------------------
import sys
sys.path.insert(2, "/opt/homebrew/lib/python3.11/site-packages")
#---------------------------------------------------------------
from streamlit_option_menu import option_menu
import pandas as pd
from annotated_text import annotated_text
from PIL import Image
#avni bhardwaj
#change in tab icon and title:
img = Image.open('Source/Frontend/logo_login.jpg')        
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
    st.title("Basic Table")
    
    # Create a sample dataframe
    data = {
        'ID': [],
        'Type': [],
        'Cost': [],
        'Time':[],
        'Severity Rating':[]
    }
    df = pd.DataFrame(data)
    
    # Display the dataframe as a table
    st.dataframe(df)        
    
def programmatic_actions():
    st.title("These are the programmatic actions")
    data2={
        'Incident ID':[],
        'Medical Emergency':[],
        'Legal Emergency':[],
        'Police Aid':[]
    }
    df2= pd.DataFrame(data2)
    st.dataframe(df2)
    
def manual_actions():
    st.title("These are the manual actions") 
    st.video("https://youtu.be/p5Gu_c_LpiA")

    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
    background-color: #ce1126;
    font-family=serif;
    color: white;
    height: 3em;
    width: 12em;
    border-radius:10px;
    border:3px solid #000000;
    font-size:20px;
    font-weight: bold;
    margin: auto;
    display: block;
    }

    div.stButton > button:hover {
	background:linear-gradient(to bottom, #ce1126 10%, #ff5a5a 100%);
	background-color:#ce1126;
    }

    div.stButton > button:active {
	position:relative;
	top:3px;
    }

</style>""", unsafe_allow_html=True)


    st.markdown('<p></p>', unsafe_allow_html = True)
    st.markdown('<p></p>', unsafe_allow_html = True)
    st.markdown('<p></p>', unsafe_allow_html = True)
    b = st.button("Police Aid")
    c = st .button("Medical aid")
    d =st.button("Legal Aid")
    if st.button("Severity Rating"):
        slider= st.slider('Provide manual rating', 0, 10,1)
        st.write("Rating is:",slider)
        
        
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
    