# class house:
#     floors = 3
#     default_color = "grey"

#     def __init__(self, len, width):
#         self.len = len
#         self.width = width
    
#     def add_floor(self):
#         self.floors += 1
#     def get_floors_count(self):
#         return self.floors
#     def get_square(self):
#         return self.len * self.width * int(self.floors)




# x, y = [int(x) for x in input().split()]

# h1 = house(x,y)
# print(h1.get_square())

# h1.add_floor()

# print(h1.get_square())

from psycopg2 import OperationalError
import psycopg2

def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database = db_name,
            user = db_user,
            password = db_password,
            host = db_host,
            port = db_port
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occured")
    return connection


def create_database(connection, query):
    connection.autocommit = True
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Query executed successfully")
    except OperationalError as e:
        print(f"The error '{e}' occured")



# connection = create_connection(
#     "postgres", "postgres", "1", "127.0.0.1", "5432"
# )

# create_database_query = "CREATE DATABASE sm_app"

#create_database(connection, create_database_query)

connection = create_connection(
    "sm_app", "postgres", "1", "127.0.0.1", "5432"
)
