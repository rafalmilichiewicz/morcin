import os
import mysql.connector
from mysql.connector import Error


def get_db_connection():
    """
    Create a MySQL database connection using environment variables.

    Environment variables expected:
        DB_HOST
        DB_USER
        DB_PASSWORD
        DB_NAME  (optional, if applicable)
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME")  # can be None if not set
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None
