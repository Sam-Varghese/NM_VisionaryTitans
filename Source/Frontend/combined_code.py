import sqlite3
import hashlib
import base64
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import re
import numpy as np
from PIL import Image
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(1, "/opt/homebrew/lib/python3.11/site-packages")
import requests
import streamlit.components.v1 as components
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_lottie import st_lottie
import json
import plotly.express as px

#change in tab icon and title:
img = Image.open('images/logo_title.jpg')
st.set_page_config(page_title="Cyber Security App", page_icon=img)

css = '''
<style>
    body {
        background-image: linear-gradient(to left, #e0429c, #cb4faf, #b25bbd, #9566c6, #766eca, #5c7cd5, #3b89db, #0095dd, #00abe6, #00c0e8, #00d4e3, #36e6db);
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

# Removed the footer:
hide_menu_style = """ 
    <style>
    footer {visibility:hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# code block for animated picture
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coder = load_lottieurl(
    "https://assets9.lottiefiles.com/packages/lf20_gxcnsfk2.json")

lottie_lock = load_lottieurl(
    "https://assets9.lottiefiles.com/packages/lf20_ndt8zfny.json")


# Security
# passlib,hashlib,bcrypt,scrypt
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password_requirements(password):
    # Check if password meets the requirements
    if len(password) < 8:
        return False

    if not re.search(r"[a-z]", password):
        return False

    if not re.search(r"[A-Z]", password):
        return False

    if not re.search(r"\W", password):
        return False
    
    return True

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()

# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',
            (username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',
            (username, password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data



#defining code for 4 main options
def cctv_footages():
    st.write("Video Player")
    video_url = r"D:\VS Code Workspace\Fair Space\GitHub\NM_VisionaryTitans\Source\Frontend\accident_footage.mp4"

    if video_url:
        st.video(video_url)
        # User input for video time
        video_time = st.number_input(
            "Select time (in seconds)", min_value=0, step=1, value=0)

        # Play button
        play_button = st.button("Play")

        # Video playback
        video_path = r"D:\VS Code Workspace\Fair Space\GitHub\NM_VisionaryTitans\Source\Frontend\accident_footage.mp4"

        # Check if play button is clicked
        if play_button:
            st.video(video_path, start_time=video_time)


def incidents_database():
    st.title("Incident Database Table")
    # Create a sample dataframe
    data = {
        'ID': [11, 12, 13, 14, 15],
        'Type': ['B', 'A', 'A', 'B', 'C'],
        'Cost': [1000, 2000, 1500, 3000, 2500],
        'Date': ['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04', '2023-07-05'],
        'Time': ['11:00', '11:30', '12:00', '12:15', '13:45'],
        'Severity Rating': [2, 1, 1, 4, 2]
    }

    df = pd.DataFrame(data)

    # Checkbox to enable/disable filtering
    enable_filter = st.checkbox("Enable Filtering")

    if enable_filter:
        # Input fields for filtering criteria
        filtered_df = dataframe_explorer(df, case=False)
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
    
def programmatic_actions():
    st.title("These are the programmatic actions")
    data2={
        'Incident ID':[],
        'Medical Emergency':[],
        'Legal Emergency':[],
        'Police Aid':[]
    }
    dataframe2= pd.DataFrame(data2)

    # Checkbox to enable/disable filtering
    enable_filter = st.checkbox("Enable Filtering")

    if enable_filter:
        filtered_df = dataframe_explorer(dataframe2, case=False)
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.dataframe(dataframe2, use_container_width=True)
    
    
def manual_actions():
    video_url = r"D:\VS Code Workspace\Fair Space\GitHub\NM_VisionaryTitans\Source\Frontend\accident_footage.mp4"
    if video_url:
        st.video(video_url)

    selected = option_menu(
        menu_title=None,
        options=['Police Aid', 'Medical Aid', 'Legal Aid', 'Severity Rating'],
        icons=['taxi-front-fill', 'hospital',
            'person-lines-fill', 'exclamation-triangle-fill'],
        default_index=0,
        orientation='horizontal'
    )
    if selected == 'Police Aid':
        st.write('Call to police')
    if selected == 'Medical Aid':
        st.write('Call to Ambulance')
    if selected == 'Legal Aid':
        st.write('Call to Lawyer')
    if selected == 'Severity Rating':
        st.slider('Check for severity ', 0, 10, 5)
        st.write('Highly Severe !')

# Main Application Page
def main_app():
    st.title("NM_VisionaryTitans")
    # Check authentication status
    if st.session_state.get("isAuthenticated"):
        st.success("Login successful!")
        # Show the rest of the application
        st_lottie(lottie_lock, width=700, height =200)
        selected = option_menu(menu_title=None,
        options=['CCTV Footages', 'Incidents Database','Programmatic Actions', 'Manual Actions', 'Report'],
        icons=['camera-fill', 'database-check','chevron-double-right', 'clock-fill','file-earmark-text'],
        default_index=0,
        orientation='horizontal')
        if selected == "CCTV Footages":
            cctv_footages()
        if selected == "Incidents Database":
            incidents_database()
        if selected == "Programmatic Actions":
            programmatic_actions()
        if selected == "Manual Actions":
            manual_actions()
        if selected == "Report":
            st.write('Analysis of past years data')
            st.write('Download')
    
    else:
        st.error("You need to log in.")



# Login Page
def login_app():

    # Add a colorful welcome message
    st.markdown(
        """
        <div style="background-color:#000f89;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">Welcome to the Cybersecurity Login App!</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    menu = ["Home", "Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown("<h1 style='font-size: 18px;'>Home</h1>",
                    unsafe_allow_html=True)
        st_lottie(lottie_coder)

    elif choice == "Login":
        st.subheader("Login Section")
        username = st.text_input("User Name")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username, check_hashes(password, hashed_pswd))

            if result:
                # Set authentication status
                st.session_state.isAuthenticated = True
                # Redirect to main app
                st.experimental_rerun()

            else:
                st.error("Invalid credentials")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        st.markdown('Password Length : >8 characters')
        st.markdown('Characters should be b/w : [A-Z] and [a-z]')
        st.markdown('Can contain underscores')
        st.markdown('Alphanumeric characters are NOT allowed')
        
        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")



# Run the appropriate app based on authentication status
if "isAuthenticated" not in st.session_state:
    login_app()
else:
    main_app()