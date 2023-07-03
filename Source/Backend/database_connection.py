import mysql.connector

# Establish a connection to the MySQL database
conn = mysql.connector.connect(
    host="localhost",    # Replace with the actual host name
    user="root",     # Replace with the actual username
    password="12345", # Replace with the actual password
)
mycursor = conn.cursor()
mycursor.execute('''create database if not exists NM_VisionaryTitans''')
conn.commit()
mycursor.execute('use NM_VisionaryTitans')
mycursor.execute('''create table if not exists IncidentsDatabase(
ID varchar(20) PRIMARY KEY,
type_of_accident varchar(200) NOT NULL,
time_of_accident datetime NOT NULL,
location varchar(100) NOT NULL,
severity int(2) NOT NULL,
Medical_Aid varchar(5) NOT NULL,
Legal_Aid varchar(5)NOT NULL,
Police_Aid varchar(5) NOT NULL
)''')

mycursor.execute('''create table if not exists ProgrammaticActions(
Incident_id varchar(20) PRIMARY KEY,
Medical_Aid varchar(5) NOT NULL,
Legal_Aid varchar(5) NOT NULL,
Police_Aid varchar(5) NOT NULL
)''')
mycursor.execute(''' create table if not exists ManualActions(
Incident_id varchar(20) UNIQUE,
Medical_Aid varchar(5) NOT NULL,
Legal_Aid varchar(5)NOT NULL,
Police_Aid varchar(5) NOT NULL)''')
conn.commit()
mycursor.execute('''insert into IncidentsDatabase values('gfr45','car crash','2023-04-22 10:34:23.55',' 41.40338 2.17403',8,'True','False','True')''')
conn.commit()
