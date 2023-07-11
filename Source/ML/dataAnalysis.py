import matplotlib.pyplot as plt
import mysql.connector
from mysql.connector import errorcode
import time

plt.figure(figsize=(15, 10))

class DatabaseConnector:
    """For interacting with MySQL database."""
    def __init__(self):
        self.username = "root"
        self.password = "root"
        self.host = "localhost"
        self.database = "NM_VisionaryTitans"
        self.connection = None
        self.rows_inserted = 0
        self.result_fetched = None

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
            create_table_query = """CREATE TABLE IF NOT EXISTS realTimeTrends (
                  id VARCHAR(255) PRIMARY KEY,
                  StartTime VARCHAR(255) NOT NULL,
                  EndTime VARCHAR(255) NOT NULL,
                  PeopleCount INT,
                  VehicleCount INT,
                  AverageSpeed FLOAT NULL
                )"""

            # Execute the table creation statement
            self.cursor.execute(create_table_query)
            self.connection.commit()
            print("Tables created successfully.")

        except mysql.connector.Error as err:
            print("An error occurred:", err)

    def updateRealTimeTrends(self, start_time, end_time, people_count, vehicle_count, avg_speed):
        
        try:
            if(avg_speed == None):
                avg_speed = "NULL"
            # Define the INSERT statement
            insert_query = """INSERT INTO realTimeTrends 
                (Id, StartTime, EndTime, PeopleCount, VehicleCount, AverageSpeed) 
                VALUES ('{}', '{}', '{}', {}, {}, {})""".format(str(uuid.uuid4()), start_time, end_time, people_count, vehicle_count, avg_speed)
            
            self.cursor.execute(insert_query)
            self.connection.commit()
            self.rows_inserted += 1
            print("Inserted {}th datapoint".format(self.rows_inserted))

        except mysql.connector.Error as err:
            print("An error occurred:", err)

    def extract_all_data(self):
        self.cursor.execute("SELECT * FROM {};".format(self.table))
        self.result_fetched = self.cursor.fetchall()

    def execute_custom_query(self, query):
        self.cursor.execute(query)
        self.result_fetched = self.cursor.fetchall()

    def calculate_time_difference(self, time1, time2):
        # Convert time strings to time objects
        time_obj1 = time.strptime(time1, "%d %B, %Y %I:%M:%S %p")
        time_obj2 = time.strptime(time2, "%d %B, %Y %I:%M:%S %p")

        # Convert time objects to seconds since epoch
        time_sec1 = time.mktime(time_obj1)
        time_sec2 = time.mktime(time_obj2)

        # Calculate the time difference in seconds
        time_diff_sec = time_sec2 - time_sec1

        # Convert the time difference to minutes and seconds
        minutes = int(time_diff_sec // 60)
        seconds = int(time_diff_sec % 60)

        # Return the time difference
        return minutes, seconds

databaseConnector = DatabaseConnector()
databaseConnector.connect()
databaseConnector.table = "cyberabad_traffic_incident4"
databaseConnector.execute_custom_query("SELECT STARTTIME, AVERAGESPEED, VEHICLECOUNT, PEOPLECOUNT FROM {};".format(databaseConnector.table))
data = databaseConnector.result_fetched

start_times = [i[0] for i in data]
average_speeds = [i[1] if i[1] != None else 0 for i in data ]
vehicle_count = [i[2] if i[2] != None else 0 for i in data ]
# people_count = [i[3] if i[3] != None else 0 for i in data ]

plt.xlabel("Time")
plt.ylabel("Count")
plt.xticks(rotation = 45*2)
plt.plot(start_times, vehicle_count, label = "Vehicle Count")
# plt.plot(start_times, people_count, label = "People Count")
# plt.plot(start_times, average_speeds, label = "Speeds")
plt.legend()
# plt.savefig("Source/ML/plots/{}_combine.png".format(databaseConnector.table))
plt.show()