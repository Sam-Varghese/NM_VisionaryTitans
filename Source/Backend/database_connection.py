import mysql.connector

# Establish a connection to the MySQL database
conn = mysql.connector.connect(
    host="localhost",    # Replace with the actual host name
    user="root",     # Replace with the actual username
    password="12345", # Replace with the actual password
)
mycursor = conn.cursor()
mycursor.execute('''create database if not exists NM_VisionaryTitans;
create table if not exists IncidentsDatabase(
ID varchar(20) PRIMARY KEY,
 type_of_accident varchar(200) NOT NULL,
 time_of_accident datetime NOT NULL,
 location varchar(100) NOT NULL,
 severity int(2) NOT NULL,
 Medical_Aid varchar(5) NOT NULL,
 Legal_Aid varchar(5)NOT NULL,
 Police_Aid varchar(5) NOT NULL
);
create table if not exists ProgrammaticActions(
Incident_id varchar(20) PRIMARY KEY,
Medical_Aid varchar(5) NOT NULL,
Legal_Aid varchar(5) NOT NULL,
Police_Aid varchar(5) NOT NULL
);
create table if not exists ManualActions(
Incident_id varchar(20) UNIQUE,
Medical_Aid varchar(5) NOT NULL,
Legal_Aid varchar(5)NOT NULL,
Police_Aid varchar(5) NOT NULL);
insert into IncidentsDatabases values('12wers,'car crash','2023-04-22 10:34:23.55',' 41.40338, 2.17403',8,'True','False');
select * from IncidentDatabases;
''',multi=True)
conn.commit()
def updateIncidentsDatabase(ID, Type, Time, Location, Severity, medicalAid, legalAid, policeAid):
    if(ID, Type, Time, Location, Severity, medicalAid, legalAid, policeAid):
        return True
    else:
        return False
    
def updateProgrammaticActions(Incident_id,Medical_Aid,Legal_Aid,Police_Aid):
    if(Incident_id,Medical_Aid,Legal_Aid,Police_Aid):
        return True
    else:
        return False

def updateManualActions(Incident_id,Medical_Aid,Legal_Aid,Police_Aid):
    if(Incident_id,Medical_Aid,Legal_Aid,Police_Aid):
        return True
    else:
        return False