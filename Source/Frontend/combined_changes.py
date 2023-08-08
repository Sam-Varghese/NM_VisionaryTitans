import sys
sys.path.insert(1, "/opt/homebrew/lib/python3.11/site-packages")
import mysql.connector
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
import requests
import streamlit.components.v1 as components
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_lottie import st_lottie
import json
import plotly.express as px
import matplotlib.pyplot as plt
import folium
import streamlit as st
import pydeck as pdk
from mysql.connector import errorcode

#change in tab icon and title:
img = Image.open('Source/Frontend/logo_title.jpg')
layout="wide"
st.set_page_config(page_title="Cyber Security App", page_icon=img, layout=layout)


# Removed the footer:
hide_menu_style = """ 
    <style>
    footer {visibility:hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

#Font style change 
css = """
<style>
body {
    font-family: cursive;;
}
</style>
"""

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
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Video Player</h3>", unsafe_allow_html=True)
    video_url = r"Source/Frontend/accident_footage.mp4"

    if video_url:
        st.video(video_url)
        # User input for video time
        video_time = st.number_input(
            "Select time (in seconds)", min_value=0, step=1, value=0)

        # Play button
        play_button = st.button("Play")

        # Video playback
        video_path = r"Source/Frontend/accident_footage.mp4"

        # Checking  play button is clicked
        if play_button:
            st.video(video_path, start_time=video_time)


def incidents_database():
    st.markdown(css, unsafe_allow_html=True)
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
    st.markdown(css, unsafe_allow_html=True)
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
    st.markdown(css, unsafe_allow_html=True)
    video_url = r"Source/Frontend/accident_footage.mp4"
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

def report(self):
        df = pd.read_csv("Source/Frontend/dataset.csv")
        st.title("Real-Time Report")
        job_filter = st.selectbox("Select the Job", pd.unique(df['job']))
        # creating a single-element container.
        placeholder = st.empty()
        # dataframe filter
        df = df[df['job'] == job_filter]
        minute_vehicle_count_df = databaseConnector.allDataToDataframe(100, "select time, count(distinct vehicleName) as totalVehicles from {} group by time;".format(self.table_name))
        coordinates_df = databaseConnector.allDataToDataframe(100, "SELECT topLeftX, topLeftY from {} limit 100;".format(self.table_name))
        avg_speed_timedf = databaseConnector.allDataToDataframe(100, "select time, AVG(speed) as avg_speed from {} group by time;".format(self.table_name))
        # near real-time / live feed simulation
        for seconds in range(200):
            df['age_new'] = df['age'] * np.random.choice(range(1, 5))
            df['balance_new'] = df['balance'] * np.random.choice(range(1, 5))
            # creating KPIs
            avg_age = np.mean(df['age_new'])
            count_married = int(df[(df["marital"] == 'married')]
                ['marital'].count() + np.random.choice(range(1, 30)))
            balance = np.mean(df['balance_new'])

            with placeholder.container():
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric(label="Incidents Count 🤕", value=round(
                avg_age), delta=round(avg_age) - 10)
                kpi2.metric(label="Accidents Count 🚗", value=int(
                count_married), delta=- 10 + count_married)
                kpi3.metric(label="Alerts Sent 🚨",
                        value=f"{round(balance,2)} ", delta=- round(balance/count_married) * 100)

                # create two columns for charts
                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    st.markdown("### Crowd vs Time")
                    fig2 = px.histogram(data_frame=avg_speed_timedf, x='time', y = "avg_speed")
                    st.write(fig2)
                with fig_col2:
                    st.markdown("### Crowd vs Time")
                    fig2 = px.histogram(data_frame=minute_vehicle_count_df, x='time', y = "totalVehicles")
                    st.write(fig2)
                print(st.columns(1))
                fig_col3 = st.columns(1)

                with fig_col3[0]:
                    st.markdown("### Live coordinates")
                    fig3 = px.scatter(coordinates_df, x = "topLeftX", y = "topLeftY")
                    st.write(fig3)
    # center on Liberty Bell, add marker
        st.markdown("###Third Chart-Location")
        m = folium.Map(location=[19.0748, 72.8856], zoom_start=16)
        folium.Marker(
        [19.0748, 72.8856], popup="Liberty Bell", tooltip="Liberty Bell"
        ).add_to(m)

# call to render Folium map in Streamlit
        st_data = st_folium(m, width=725)
        st.markdown("### Fourth Chart-Area Analysis")
        chart_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])

        st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=19.0748,
        longitude=72.8856,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=chart_data,
           get_position='[lon, lat]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],
))

# Main Application Page
def main_app():
        st.markdown("<h2 style='text-align: center ;'>NM_VisionaryTitans</h2>", unsafe_allow_html=True)
        
        # Check authentication status
        if st.session_state.get("isAuthenticated"):
            selected = option_menu(menu_title=None,
            options=['CCTV Footages', 'Incidents Database','Programmatic Actions', 'Manual Actions', 'Report'],
            icons=['camera-fill', 'database-check','chevron-double-right', 'clock-fill','file-earmark-text'],
            default_index=0,
            orientation='horizontal')
            # Show the rest of the application
            if selected == "CCTV Footages":
                cctv_footages()
            if selected == "Incidents Database":
                incidents_database()
            if selected == "Programmatic Actions":
                programmatic_actions()
            if selected == "Manual Actions":
                manual_actions()
            if selected == "Report":
                report()
    
        else:
            st.error("You need to log in.")

# Login Page
def login_app():
    st.markdown(css, unsafe_allow_html=True)

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
        st.markdown("<h2 style='text-align: center;'>HOME</h2>", unsafe_allow_html=True)
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