import mysql.connector
import pandas as pd
import sqlite3
import hashlib
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import re
from PIL import Image
import sys
sys.path.insert(1, "/opt/homebrew/lib/python3.11/site-packages")
import requests
from streamlit_extras.dataframe_explorer import dataframe_explorer
from mysql.connector import errorcode
from streamlit_lottie import st_lottie
import plotly.express as px
from rough import generate_frames

# Class for dealing with database connections
class DatabaseConnector:
    """For interacting with MySQL database."""
    def __init__(self):
        self.username = "root"
        self.password = "root"
        self.host = "localhost"
        self.database = "NM_VisionaryTitans"
        self.connection = None
        self.gen_rows_inserted = 0
        self.sp_rows_inserted = 0
        self.generalInfoTable = None
        self.specificInfoTable = None

    def connect(self):
        """Establishes python and MySQL connection, creates the database if it doesn't exist."""
        try:
            # Establish a connection to the MySQL server
            self.connection = mysql.connector.connect(
                user=self.username,
                password=self.password,
                host=self.host,
            )
            self.cursor = self.connection.cursor()

            self.cursor.execute("CREATE DATABASE IF NOT EXISTS {}".format(self.database))
            self.connection.commit()

            self.cursor.execute("USE {}".format(self.database))

            print("Connected to the database.")

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Error: Access denied. Check your username and password.")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Error: The specified database does not exist.")
            else:
                print("An error occurred:", err)

    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("Disconnected from the database.")

    def create_tables(self):
        try:

            # Define the table creation statement
            create_table1_query = """CREATE TABLE IF NOT EXISTS {} (
                  id INT AUTO_INCREMENT PRIMARY KEY,
                  StartTime VARCHAR(255) NOT NULL,
                  EndTime VARCHAR(255) NOT NULL,
                  PeopleCount INT,
                  VehicleCount INT,
                  AverageSpeed FLOAT NULL
                )""".format(self.generalInfoTable)
            
            create_table2_query = """CREATE TABLE IF NOT EXISTS {} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                time VARCHAR(255) NOT NULL,
                vehicleName VARCHAR(255) NOT NULL,
                topLeftX FLOAT NOT NULL,
                topLeftY FLOAT NOT NULL,
                bottomRightX FLOAT NOT NULL,
                bottomRightY FLOAT NOT NULL,
                speed FLOAT NULL
            )""".format(self.specificInfoTable)

            # Execute the table creation statements
            self.cursor.execute(create_table1_query)
            self.cursor.execute(create_table2_query)

            self.connection.commit()
            print("Tables created successfully.")


        except mysql.connector.Error as err:
            print("An error occurred:", err)

    def updateGenTable(self, start_time, end_time, people_count, vehicle_count, avg_speed):
        try:
            if(avg_speed == None):
                avg_speed = "NULL"
            # Define the INSERT statement, here UUID has been used because nor vehicle/ person't ID will be suitable to define a particular time instant
            insert_query = """INSERT INTO {} 
                (StartTime, EndTime, PeopleCount, VehicleCount, AverageSpeed) 
                VALUES ('{}', '{}', {}, {}, {})""".format(self.generalInfoTable, start_time, end_time, people_count, vehicle_count, avg_speed)
            
            self.cursor.execute(insert_query)
            self.connection.commit()
            self.gen_rows_inserted += 1
            # print("Inserted {}th datapoint of general data".format(self.gen_rows_inserted))

        except mysql.connector.Error as err:
            print("An error occurred:", err)

    def updateSpTable(self, time, vehicle_name, topLeftX, topLeftY, bottomRightX, bottomRightY, speed):
        try:
            insert_query = """INSERT INTO {} 
                (time, vehicleName, topLeftX, topLeftY, bottomRightX, bottomRightY, speed) 
                VALUES ('{}', '{}', {}, {}, {}, {}, {})""".format(self.specificInfoTable, time, vehicle_name, topLeftX, topLeftY, bottomRightX, bottomRightY, speed)
            
            self.cursor.execute(insert_query)
            self.connection.commit()
            self.sp_rows_inserted += 1
            # print("Inserted {}th datapoint of specific data".format(self.sp_rows_inserted))

        except mysql.connector.Error as err:
            print("An error occurred at function updateSpTable:", err)

    def clearTables(self):
        self.cursor.execute("DROP TABLE IF EXISTS {}".format(self.generalInfoTable))
        self.cursor.execute("DROP TABLE IF EXISTS {}".format(self.specificInfoTable))

        self.connection.commit()
        print("All tables cleared.")

    def allDataToDataframe(self, limit: int, customQuery = None):
        if (customQuery == None):
            query = "SELECT * FROM {} LIMIT {};".format(self.specificInfoTable, limit)
        else:
            query = customQuery
        df = pd.read_sql_query(query, self.connection)
        return df
    
class Frontend:
    def __init__(self, databaseConnector: DatabaseConnector):
        self.css = '''
            <style>
                body {
                    background-image: linear-gradient(to left, #e0429c, #cb4faf, #b25bbd, #9566c6, #766eca, #5c7cd5, #3b89db, #0095dd, #00abe6, #00c0e8, #00d4e3, #36e6db);
                }
            </style>
            '''
        self.logo_img = Image.open('D:\VS Code Workspace\Fair Space\GitHub\NM_VisionaryTitans\images\logo_title.jpg')
        self.hide_menu_style = """ 
            <style>
            footer {visibility:hidden;}
            </style>
            """
        st.set_page_config(page_title="Cyber Security App", page_icon=self.logo_img)
        st.markdown(self.css, unsafe_allow_html=True)
        # code block for animated picture

        self.lottie_coder = self.load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_gxcnsfk2.json")

        self.lottie_lock = self.load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_ndt8zfny.json")
        # DB Management
        self.conn = sqlite3.connect('data.db')
        self.c = self.conn.cursor()
        self.databaseConnector = databaseConnector
        self.table_name = databaseConnector.specificInfoTable

    def check_password_requirements(self, password):
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

    def check_hashes(self, password, hashed_text):
        if self.make_hashes(password) == hashed_text:
            return hashed_text
        return False

    def load_lottieurl(self, url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    # Security
    # passlib,hashlib,bcrypt,scrypt
    def make_hashes(self, password):
        return hashlib.sha256(str.encode(password)).hexdigest()

    def create_usertable(self):
        self.c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

    def add_userdata(self, username, password):
        self.c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',
                (username, password))
        self.conn.commit()

    def login_user(self, username, password):
        self.c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',
                (username, password))
        data = self.c.fetchall()
        return data

    def view_all_users(self):
        self.c.execute('SELECT * FROM userstable')
        data = self.c.fetchall()
        return data

    #defining code for 4 main options
    def cctv_footages(self):
        st.markdown(self.css, unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Video Player</h3>", unsafe_allow_html=True)

        video_placeholder = st.empty()
        # Loop to continuously read frames and update the displayed image
        for frame in generate_frames():
            # Display the frame on Streamlit
            video_placeholder.image(frame, channels="RGB", use_column_width=True)



    def incidents_database(self):
        st.markdown(self.css, unsafe_allow_html=True)
        st.title("Incident Database")
        
        df = self.databaseConnector.allDataToDataframe(10)

        # Checkbox to enable/disable filtering
        enable_filter = st.checkbox("Enable Filtering")

        if enable_filter:
            # Input fields for filtering criteria
            filtered_df = dataframe_explorer(df, case=False)
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
        
    def programmatic_actions(self):
        st.markdown(self.css, unsafe_allow_html=True)
        st.title("Programmatic Actions")
        data2 = {
    'Incident ID': [3489, 5712, 9267, 1823, 6894, 2405, 5038, 7761, 6382, 4156,
                    2957, 1640, 8319, 5682, 9537, 7148, 2901, 8745, 6678, 4271,
                    1594, 7427, 6120, 3953, 8766, 2239, 4012, 5485, 9638, 1311,
                    8874, 2567, 6790, 5703, 8436, 2189, 3022, 4855, 7538, 6291,
                    7844, 3607, 9470, 1053, 5126, 9869, 7652, 5395, 8868, 2541],
    'Time': [
        "Tue Jan 10 06:42:15 2023", "Wed Feb 15 12:20:59 2023", "Fri Mar 24 08:15:30 2023",
        "Sat Apr 01 23:58:07 2023", "Mon May 08 14:07:42 2023", "Tue Jun 13 19:34:50 2023",
        "Thu Jul 20 09:10:25 2023", "Sat Aug 05 17:55:18 2023", "Mon Sep 11 03:27:33 2023",
        "Wed Oct 18 22:45:09 2023", "Fri Nov 24 16:38:51 2023", "Sun Dec 31 07:53:22 2023",
        "Tue Jan 09 14:16:37 2023", "Thu Feb 22 05:31:05 2023", "Sat Mar 11 21:09:44 2023",
        "Mon Apr 17 10:36:29 2023", "Wed May 24 03:50:58 2023", "Fri Jun 30 18:12:40 2023",
        "Sun Aug 13 07:22:17 2023", "Tue Sep 26 16:04:02 2023", "Thu Nov 02 23:27:50 2023",
        "Sat Dec 09 12:53:31 2023", "Mon Jan 15 01:08:14 2023", "Wed Feb 21 08:31:47 2023",
        "Fri Mar 30 19:47:03 2023", "Sun May 06 09:55:25 2023", "Tue Jun 12 01:14:37 2023",
        "Thu Jul 19 17:40:12 2023", "Sat Aug 25 03:02:55 2023", "Mon Oct 02 14:18:20 2023",
        "Wed Nov 08 06:31:45 2023", "Fri Dec 15 22:49:28 2023", "Sun Jan 21 13:01:50 2023",
        "Tue Feb 27 00:22:06 2023", "Thu Apr 05 08:38:33 2023", "Sat May 12 17:50:59 2023",
        "Mon Jun 18 04:07:24 2023", "Wed Jul 26 12:25:51 2023", "Fri Sep 01 20:42:19 2023",
        "Sun Oct 08 04:56:45 2023", "Tue Nov 14 15:09:10 2023", "Thu Dec 21 03:21:35 2023",
        "Sat Jan 27 17:33:01 2023", "Mon Mar 06 07:44:26 2023", "Wed Apr 12 19:55:52 2023",
        "Fri May 19 10:08:18 2023", "Sun Jun 25 00:19:43 2023", "Tue Aug 01 12:31:09 2023",
        "Thu Sep 07 02:42:35 2023", "Sat Oct 14 14:54:00 2023", "Mon Nov 20 05:05:26 2023",
        "Wed Dec 27 19:16:52 2023"
    ],
    'Medical Emergency': ["True", "False", "True", "False", "True", "False", "True", "False", "True", "False","True", "False", "True", "False", "True", "False", "True", "False", "True", "False","True", "False", "True", "False", "True", "False", "True", "False", "True", "False","True", "False", "True", "False", "True", "False", "True", "False", "True", "False","True", "False", "True", "False", "True", "False", "True", "False", "True", "False"],
    'Legal Emergency': ["False", "True", "False", "True", "False", "True", "False", "True", "False", "True","False", "True", "False", "True", "False", "True", "False", "True", "False", "True","False", "True", "False", "True", "False", "True", "False", "True", "False", "True",
"False", "True", "False", "True", "False", "True", "False", "True", "False", "True",
"False", "True", "False", "True", "False", "True", "False", "True", "False", "True"],
    'Police Aid': ["True", "False", "True", "False", "True", "False", "True", "False", "True", "False","True", "False", "True", "False", "True", "False", "True", "False", "True", "False","True", "False", "True", "False", "True", "False", "True", "False", "True", "False","True", "False", "True", "False", "True", "False", "True", "False", "True", "False","True", "False", "True", "False", "True", "False", "True", "False", "True", "False"]
}

        dataframe2= pd.DataFrame(data2)

        # Checkbox to enable/disable filtering
        enable_filter = st.checkbox("Enable Filtering")

        if enable_filter:
            filtered_df = dataframe_explorer(dataframe2, case=False)
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(dataframe2, use_container_width=True)

    def manual_actions(self):
        st.markdown(self.css, unsafe_allow_html=True)
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
            st.write('Call police officials')
            # Put the function
            st.button("Sam")
        if selected == 'Medical Aid':
            st.write('Request ambulance')
        if selected == 'Legal Aid':
            st.write('Request lawyers')
        if selected == 'Severity Rating':
            st.slider('Check for severity ', 0, 10, 5)
            st.write('Highly Severe!')

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

    # Main Application Page
    def main_app(self):
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
                self.cctv_footages()
            if selected == "Incidents Database":
                self.incidents_database()
            if selected == "Programmatic Actions":
                self.programmatic_actions()
            if selected == "Manual Actions":
                self.manual_actions()
            if selected == "Report":
                self.report()
    
        else:
            st.error("You need to log in.")

    # Login Page
    def login_app(self):

        st.markdown(self.css, unsafe_allow_html=True)

        # Add a colorful welcome message
        st.markdown(
            """
            <div style="background-color:#000f89;padding:10px;border-radius:10px">
            <h1 style="color:white;text-align:center;">NM_VisionaryTitans!</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        menu = ["Home", "Login", "SignUp"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Home":
            st.markdown("<h2 style='text-align: center;'>Home</h2>", unsafe_allow_html=True)
            st_lottie(self.lottie_coder)

        elif choice == "Login":
            st.subheader("Login Section")
            username = st.text_input("User Name")
            password = st.text_input("Password", type='password')
            if st.button("Login"):
                self.create_usertable()
                hashed_pswd = self.make_hashes(password)
                result = self.login_user(username, self.check_hashes(password, hashed_pswd))

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
                self.create_usertable()
                self.add_userdata(new_user, self.make_hashes(new_password))
                st.success("You have successfully created a valid account")
                st.info("Kindly login to continue")

    def start_application(self):
        # Run the appropriate app based on authentication status
        if "isAuthenticated" not in st.session_state:
            self.login_app()
        else:
            self.main_app()

databaseConnector = DatabaseConnector()
databaseConnector.connect()
video_path = "Source/ML/accidents/pakistan_accident_1.mp4"

name = video_path.split("/")[-1].split(".")[0]

databaseConnector.generalInfoTable = "gen_" + name
databaseConnector.specificInfoTable = "sp_" + name

frontend = Frontend(databaseConnector)
frontend.start_application()